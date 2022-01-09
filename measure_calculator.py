import pandas as pd
import dask as dd
import re
import datetime
from typing import List, Callable
import numpy as np
import os
import sys
import datetime
from pytz import timezone
import math

# below are place holders for now (until we get actual layer 2 tables for computation)
student_id = 'canvas_user_id'
course_code = "canvas_course_id"
discussions_file = 'toy_disc.csv'
assignments_file = 'toy_asgmt.csv'
quizzes_file = 'toy_quiz.csv'
clickAction_file = 'clickActionsample.csv'
session_file = 'toy_study_session.csv'
groupMem_file = 'toy_group_membership.csv'
enroll_file = 'toy_enrollment.csv'
directMess_file = 'toy_direct_message.csv'
modProg_file = 'toy_module_progression.csv'

# # column names for columns where time stamps need to be converted to US/Pacific
# discuss_dates = ['created_at', 'updated_at', 'deleted_at', 'discussion_topic_posted_at',
#                  'discussion_topic_delayed_post_at']
# assign_dates = ['created_at', 'submitted_at', 'submission_wiki_created_at', 'submission_wiki_updated_at', 'graded_at',
#                 'posted_at', 'assignment_created_at',
#                 'assignment_unlock_at', 'assignment_lock_at', 'assignment_due_at', 'assignment_created_at_overridden',
#                 'assignment_unlock_at_overridden', 'assignment_lock_at_overridden',
#                 'assignment_due_at_overridden']
# quiz_dates = ['quiz_submission_created_at', 'quiz_submission_updated_at', 'quiz_submission_started_at',
#               'quiz_submission_finished_at',
#               'quiz_published_at', 'quiz_submission_due_at']
# click_dates = ['timestamp']
# session_dates = ['start_timestamp', 'end_timestamp']
# group_dates = ['membership_created_at', 'membership_updated_at', 'group_created_at', 'group_updated_at',
#                'group_deleted_at']
# enroll_dates = ['enrollment_created_at', 'enrollment_updated_at']
# directMess_dates = ['created_at']
# modProg_dates = ['created_at', 'completed_at', 'evaluated_at', 'module_created_at', 'module_deleted_at',
#                  'module_unlock_at']
# date_dates = ['startdate', 'enddate']
#
# # used for setup when reading the files
# parse_dates = [discuss_dates, assign_dates, quiz_dates, click_dates, session_dates, group_dates, enroll_dates,
#                directMess_dates, modProg_dates]

files = [discussions_file, assignments_file, quizzes_file, clickAction_file, session_file, groupMem_file, enroll_file,
         directMess_file, modProg_file]


# measurs groupby (student_id,course_code)
class MeasureCalculator:
    def __init__(self, files):
        '''
        df[0] Discussion table
        df[1] Assignment table
        df[2] Quiz table
        df[3] Click Action table
        df[4] Session table
        df[5] Group Membership table
        df[6] Enrollment table
        df[7] Direct Message table
        df[8] Module Progression table
        df[9] Date (start/end term dates)
        '''

        self.dfs = {}
        self.__read_files(files)

        # note:
        # course_code == course_id,user_id == canvas_id
        self.dfs['stcs_dt'] = self.enrollment.groupby(['canvas_user_id', 'canvas_course_id'])[
            ['start_date', 'end_date']].first().reset_index()

        # setup result dataframe to return
        self.result = self.enrollment[
            ['enrollment_id', 'canvas_user_id', 'canvas_course_id', 'sis_course_code', 'canvas_course_section_id',
             'sis_course_section_code', 'enrollment_type']]

        # convert timestamps of all tables to US/Eastern time
        for dt_name in self.dfs:
            for col in self.dfs[dt_name].filter(regex='(_at|timestamp)$').columns:
                self.dfs[dt_name][col] = pd.to_datetime(self.dfs[dt_name][col], utc=True, infer_datetime_format=
                True, errors='coerce').dt.tz_convert('US/Eastern')

            print(dt_name, 'timestamps converted.')

        # setup all tables to refernce for term/date/week calculations (merge with start/term table)
        for dt_name in self.dfs():
            self.dfs[dt_name] = self.df[dt_name].merge(self.df['stcs_dt'], how='inner')
            for col in ["start_date", "end_date"]:
                self.dfs[dt_name][col] = pd.to_datetime(self.dfs[dt_name][col]).dt.tz_localize('US/Eastern')

            print('Course dates merged with ', dt_name, 'converted.')

        self.discussion = self.dfs[discussions_file]
        self.assignments = self.dfs[assignments_file]
        self.quiz = self.dfs[quizzes_file]
        self.clicks = self.dfs[clickAction_file]
        self.session = self.dfs[session_file]
        self.group_mem = self.dfs[groupMem_file]
        self.enrollment = self.dfs[enroll_file]
        self.direct_message = self.dfs[directMess_file]
        self.module = self.dfs[modProg_file]

        # set a cap of 5 hours for any session longer than 5 hours
        self.session.loc[self.session['time_spent'] > 18000, 'time_spent'] = 18000  # 5 hours in seconds

        # setup table to refernce for term/date/week calculations (merge above table with start/term table through ("canvas_course_id"))
        # USE self.dt throughout measures (dataframe that is cleaned by applicable constraints)
        self.dt = self.session.merge(self.df['stcs_dt'], how='inner')
        for col in ["start_date", "end_date"]:
            self.dt[col] = pd.to_datetime(self.dt[col]).dt.tz_localize('US/Eastern')
        print('Course dates merged with study session converted.')

        # set up a date table to keep track of when courses start/end, total weeks, total days, term date
        self.dt['Total_days'] = (self.dt.enddate - self.dt.startdate) / np.timedelta64(1, 'D')
        self.dt['Total_weeks'] = ((self.dt.enddate - self.dt.startdate) / np.timedelta64(1, 'W')).apply(
            np.ceil)  # if weeks is not whole, rounds up
        self.dt['half_date'] = self.dt['startdate'] + (self.dt['Total_weeks'] / 2).apply(np.ceil).apply(
            lambda x: pd.Timedelta(x, unit='W'))
        self.dt['start_week_num'] = self.dt['startdate'].dt.week
        self.dt['end_week_num'] = self.dt['enddate'].dt.week

        # dataframe that is used for first quarter measure calculations
        self.d_qrt1 = self.dt
        self.d_qrt1["first"] = self.d_qrt1['start_date'] + (self.d_qrt1['Total_weeks'] / 4).apply(np.ceil).apply(
            lambda x: pd.Timedelta(x, unit='W'))  # first quarter date
        self.d_qrt1 = self.__loc(self.d_qrt1, lambda df: (
        (df['start_timestamp'] <= df['first'])))  # sessions that are within the first quarter

        # clean assignment table to only have rows where created_at is in between start_date AND end_date
        self.assignments = self.__loc(self.assignments,
                                           lambda df: (df['submission_created_at'] >= df['start_date']) & (
                                                   df['submission_created_at'] <= df['end_date']))
        self.assignments['Total_days'] = (self.assignments['end_date'] - self.assignments[
            'start_date']) / np.timedelta64(1, 'D')
        self.assignments['Total_weeks'] = ((self.assignments['end_date'] - self.assignments[
            'start_date']) / np.timedelta64(1, 'W')).apply(np.ceil)  # if total weeks is not whole, rounds up

        # assignment dataframe that is used for first quarter measure calculations
        self.d_assgn_qrt1 = self.assignments
        self.d_assgn_qrt1["first"] = self.d_assgn_qrt1['start_date'] + (self.d_assgn_qrt1['Total_weeks'] / 4).apply(
            np.ceil).apply(lambda x: pd.Timedelta(x, unit='W'))  # first quarter date
        self.d_assgn_qrt1 = self.__loc(self.d_assgn_qrt1, lambda df: (
        (df['submission_created_at'] <= df['first'])))  # sessions that are within the first quarter

        # clean table self.dt_assub to only have rows where created_at is in between start_date AND end_date
        self.discussion = self.__loc(self.discussion,
                                           lambda df: (df['discussion_post_created_at'] >= df['start_date']) & (
                                                   df['discussion_post_created_at'] <= df['end_date']))
        self.discussion['Total_days'] = (self.discussion['end_date'] - self.discussion[
            'start_date']) / np.timedelta64(1, 'D')
        self.discussion['Total_weeks'] = ((self.discussion['end_date'] - self.discussion[
            'start_date']) / np.timedelta64(1, 'W')).apply(np.ceil)  # if total weeks is not whole, rounds up

        # discussion dataframe that is used for first quarter measure calculations
        self.d_disc_qrt1 = self.discussion
        self.d_disc_qrt1["first"] = self.d_disc_qrt1['start_date'] + (self.d_disc_qrt1['Total_weeks'] / 4).apply(
            np.ceil).apply(lambda x: pd.Timedelta(x, unit='W'))  # first quarter date
        self.d_disc_qrt1 = self.__loc(self.d_disc_qrt1, lambda df: (
        (df['discussion_post_created_at'] <= df['first'])))  # sessions that are within the first quarter

        # clean quiz table to only have rows where created_at is in between start_date AND end_date
        self.quiz = self.__loc(self.quiz,
                               lambda df: (df['quiz_submission_created_at'] >= df['start_date']) &
                                          df['quiz_submission_created_at'] <= df['end_date'])

        self.quiz['Total_days'] = (self.quiz['end_date'] - self.quiz[
            'start_date']) / np.timedelta64(1, 'D')
        self.quiz['Total_weeks'] = ((self.quiz['end_date'] - self.quiz[
            'start_date']) / np.timedelta64(1, 'W')).apply(np.ceil)  # if total weeks is not whole, rounds up


        # set up replies vs posts table for easy measure calculation
        self.replies = self.discussion.dropna(
            subset=['parent_discussion_post_id'])  # if cell has value means it is a REPLYto a post
        self.posts = self.__loc(self.discussion, lambda df: df[
            'parent_discussion_post_id'].isna())  # if cell doesn't have a value means it's post

    # used within class only, loads appropriate files into list dfs; reads files into a pandas dataframe
    def __read_files(self, filenames: List[str]) -> None:
        # something to include in read_csv args  (date_parser = lambda col : pd.to_datetime(col, utc=True,infer_datetime_format=True), parse_dates = parse_dates)
        for i in range(len(filenames)):
            df = pd.read_csv(os.path.join(os.getcwd(), "src", filenames[i]),
                             parse_dates=parse_dates[i],
                             date_parser=lambda col: pd.to_datetime(col, utc=True,
                                                                    infer_datetime_format=True).dt.tz_convert(
                                 'US/Pacific')
                             )

            self.dfs[filenames[i]] = df

    # used within class only, determines the count of applicable table & function
    def __count(self, table: pd, f=None) -> pd:
        if (f is not None):
            return table.loc[f].groupby([student_id, course_code]).size()
        return table.groupby([student_id, course_code]).size()

    def __sum(self, table: pd, col: str, f=None) -> pd:
        if (f is not None):
            return table.loc[f].groupby([student_id, course_code])[col].sum()
        return table.groupby([student_id, course_code])[col].sum()

    def __loc(self, table: pd, f: Callable) -> pd:
        return table.loc[f]

    '''
    Below are measures, should follow the format:
        #Measure ID, Measure Definition
        def variable_name();
            ....
            self.result = self.result.merge(df, how = 'outer', on = [student_id,course_code])

    Measure ID, Definiton and variable name come from: https://docs.google.com/spreadsheets/d/1iHp0OOWCI__to-CTSGlnPH3FtEKOYTXbPn9J21frkdY/edit?ts=608733ae#gid=1604602785 
    '''

    # M1, # of online sessions
    def session_cnt(self):
        df = self.__count(self.session).reset_index(name='session_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M2, total time spent online
    def time(self):
        df = self.dt.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['time_spent'].sum().rename(
            columns={
                'time_spent': 'time'})
        df['time'] = df.tot_time.div(60)  # convert seconds to minutes

        # merge df to result
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    # M3, # discussion messages posted
    def created_discussion_post_cnt(self):
        df = self.__count(self.discussion).reset_index(name='created_discussion_post_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M4, # reply discussion messages posted
    def created_discussion_reply_cnt(self):

        # if depth is > 1, means its a reply
        df = self.__count(self.__loc(self.discussion, lambda df: df['depth'] > 1)).reset_index(
            name='created_discussion_reply_cnt')

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M5, # visits to grades/gradebook
    def grade_summary_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'] == 113).reset_index(
            name='grade_summary_view_cnt')  # id that corresponds to 'Click on "Grades"'
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M6, # files viewed
    def viewed_file_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'] == 191).reset_index(
            name='viewed_file_cnt')  # ids that corresponds to preview a file
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M7, # of assignments submitted
    def submitted_assignment_cnt(self):
        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = self.__count(df).reset_index(name='submitted_assignment_cnt')

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M8, Avg time spent per day (min)
    def avg_time_per_day(self):
        df = self.result[[student_id, course_code, 'tot_time']].merge(self.dt[[student_id, course_code, 'Total_days']],
                                                                      how='inner', on=[student_id, course_code])

        df["avg_time_per_day"] = (df.tot_time) / df.Total_days
        self.result = self.result.merge(df[[student_id, course_code, "avg_time_per_day"]], how='outer',
                                        on=[student_id, course_code])
        # return df

    # M9, # late assignment submissions (first half term)
    def late_assignment_submission_cnt_halfterm(self):
        submitted_column = 'submitted_at'
        due_column = 'assignment_due_at'

        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = self.__count(df,
                          lambda df: df[submitted_column] > df[due_column] and df[due_column] > half_term).reset_index(
            name='late_assignment_submission_cnt_halfterm')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M10, # assignment submission 6am-6pm
    def assignment_submitted_count_per_day(self):
        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = df[df['submitted_at'].dt.hour.between(6, 18)]
        df = self.__count(df).reset_index(name='assignment_submitted_count_per_day')

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M11, # assignment submission 6pm-12am
    def submitted_assignments_pm(self):
        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour > 18)]
        df = self.__count(df).reset_index(name='assignment_submitted_pm_count')

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M12, # assignment submission 12am-6am
    def sassignment_submitted_am_count(self):
        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour < 6)]
        df = self.__count(df).reset_index(name='assignment_submitted_am_count')

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M13, Avg session duration
    def avg_session_duration(self):
        self.result['avg_session_duration'] = self.result['time'] / self.result['session_cnt']
        # return df

    # M14, # on-time assignment submissions
    def ontime_assignment_submission_cnt(self):
        submitted_column = 'submitted_at'
        due_column = 'assignment_due_at'

        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = self.__count(df, lambda df: df[submitted_column] <= df[due_column]).reset_index(
            name='ontime_assignment_submission_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M15, # late assignment submissions (full term)
    def late_assignment_submission_cnt(self):
        submitted_column = 'submitted_at'
        due_column = 'assignment_due_at'

        df = self.assignments[self.assignments['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
        df = self.__count(df, lambda df: df[submitted_column] > df[due_column]).reset_index(
            name='late_assignment_submission_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M16, time spent on quizzes
    def quiz_time(self):
        started = 'quiz_submission_started_at'
        submitted = 'quiz_submission_finished_at'

        df = self.quiz.group_by([student_id, course_code]).apply(
            lambda df: df[submited] - df[started]).sum().reset_index(name='quiz_time')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M17, # of mail messages read (refer to inbox of canvas)
    '''
    41 == Click/show a conversation in Inbox
    '''

    def viewed_inbox_message_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'] == 41).reset_index('viewed_inbox_message_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M18, # mail messages sent (refer to inbox of canvas)
    def sent_inbox_message_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'] == 281).reset_index('sent_inbox_message_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M19, # uses of "search" function
    '''
    10 == Search for an announcement (Specify text)
    55/56 == Search for files
    '''

    def search_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([10, 55, 56])).reset_index('search_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M20, # visits to module progress
    '''
    67/68 == check module progress (along with click modules)
    '''

    def module_summary_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([67, 68])).reset_index('module_summary_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M21, # visits to inbox
    '''
    38 == Click Inbox
    40 == Click "Inbox" (api)
    '''

    def chat_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([38, 40])).reset_index('chat_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M22, # assignment views
    '''
    16 == Click into a specific assignment
    73 == Click on a module item that is an assignment
    88	== Click an assignment
    89	== Click an assignment (load prereq)
    90	== Click/Load an assignment (api)
    91	== click/load an assignment (api)
    92	== Click an assignment
    93	== Click an assignment from module main page
    117	== Click an assignment (enter the submission details) from grades page
    '''

    def assignment_view_cnt(self):
        df = self.__count(self.clicks,
                          lambda df: df['action_code'].isin([16, 73, 88, 90, 91, 92, 93, 117])).reset_index(
            'assignment_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M23, # discussion views
    '''
    26	== View the new discussion after creating
    29	== Click on a discussion post
    225	== view a discussion (api)

    '''

    def discussion_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([26, 29, 225])).reset_index(
            'discussion_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M24, Time spent on posting discussion messages
    def discussion_post_creating_time(self):
        '''
        can use click table
        only look at actions where we click reply and post
        sort these actions by timestamp for each student/course
            dictionary: student/course(key) & list of actions sorted (value)
        calcucate time difference from each of these
        '''

        '''
        get clicks where we only click reply and post to 
        32 == post a reply
        34 == Click "Reply" to an entry(reply) in a discussion
        227	== clicked reply to a discussion post
        230	== Click reply buton on a reply in discussion post in group
        '''
        df = self.clicks['action_code'].isin([32, 34, 227, 230])
        sorted_actions = df.groupby([student_id, course_code])['timestamp'].apply(
            lambda x: x.sort_values(['start_timestamp']))
        actions_dict = sorted_actions['timestamp'].apply(lambda x: x.values.tolist()).to_dict()

        sorted_actions.to_csv('test.csv')
        for k, v in actions_dict.items():
            print('this is the key:', k, ' and this is the value: ', v)

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M25, Longest period of inactivity (h)
    def max_session_gap(self):
        # start and end dates
        term = self.dt.drop_duplicates(["term_code", "course_id", "canvas_id", "startDate", "endDate"])[
            ["term_code", "course_id", "canvas_id", "startDate", "endDate"]]

        first = self.dt.groupby(["term_code", "course_id", "canvas_id"])[["start_timestamp", "end_timestamp"]].agg(
            {'start_timestamp': np.min, 'end_timestamp': np.max})
        first = first.merge(term, how='outer', on=["term_code", "course_id", "canvas_id"])
        first['start_inac'] = first['start_timestamp'] - first[
            'startDate']  # gets the inactivity between first start timestamp and startDate
        first['end_inac'] = first['endDate'] - first[
            'end_timestamp']  # gets the inactivity between last end timestamp and endDate

        # sorts the dataframe groupby ("term_code","course_id","canvas_id"), ascending order of start_timestamps
        times = self.dt.groupby(["term_code", "course_id", "canvas_id"])[["start_timestamp", "end_timestamp"]].apply(
            lambda x: x.sort_values(['start_timestamp'])).reset_index()
        times['inactivity'] = times['start_timestamp'] - times['end_timestamp'].shift(1)

        # merges the inactivity between first inactivity and last inactvity
        df = pd.concat([times[["term_code", "course_id", "canvas_id", 'inactivity']],
                        first[["term_code", "course_id", "canvas_id", "start_inac"]].rename(
                            columns={'start_inac': 'inactivity'})], ignore_index=True)
        df = pd.concat(
            [df, first[["term_code", "course_id", "canvas_id", "end_inac"]].rename(columns={'end_inac': 'inactivity'})],
            ignore_index=True)

        # gets the longest inactvity and converts to minutes
        df = df.groupby(["term_code", "course_id", "canvas_id"])['inactivity'].max().apply(
            lambda x: (x.total_seconds()) / 60).reset_index().rename(
            columns={'inactivity': 'max_session_gap'})

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M26, # Accesses to dashboard
    def dashboard_visit_count(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([26, 29, 225])).reset_index(
            'dashboard_visit_count')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    '''
    76 == Download a module item that is a file
    101 == Download the file attached in comment
    102 == Download Submission
    109 == Download the file attached in comment in assignment submission details page
    114 == Download file in an expanded comment in grades page
    212 == Download a file in group
    '''

    # M27, Download documents or files
    def file_download_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([76, 101, 102, 109, 114, 212])).reset_index(
            'file_download_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M74, Average duration of sessions with assignment accesses (h)
    def avg_session_duration_with_assignment_view(self):
        '''
        get clicks where its and assignment access

        get the session id's applicable to these clicks --> store in dict key: student id & course id, value: list of session ids
            can maybe make another dictonary where key: student id, course id, & session id, value: duration

        calculate the avg duration of these sessions --> add all session durations and divide by session count(length of list)
        '''

        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M75, Avg # actions per session in sessions with assignment accesses
    def avg_action_cnt_per_session_with_assignment_view(self):
        '''
        get clicks where its and assignment access

        get the session id's applicable to these clicks --> store in dict key: student id & course id, value: list of session ids
            can maybe make another dictonary where key: student id, course id, & session id, value: total clicks

        calculate the avg actions of these sessions --> add all sessions clicks and divide by session count(length of list)
        '''
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    '''
    84 == Click on "Syllabus" on the left panel
    123 == Click the course link that the students take (enter the syllabus page) in people page
    '''

    # M38, # of course outline views
    def syllabus_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([84, 123])).reset_index('syllabus_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    '''
    190	== Upload a file
    '''

    # M39, # times uploading files
    def upload_count(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([190])).reset_index('upload_count')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

    # M40, logins (first half term) --> should be equal to # of sessions during first half of term
    def session_cnt_halfterm(self):
        df = self.dt
        df['half_date'] = df['start_date'] + (df['Total_weeks'] / 2).apply(np.ceil).apply(lambda x: pd.Timedelta(x,
                                                                                                                 unit='W'))  # takes into account if weeks are not even (rounds up to neart whole week)

        df = self.__loc(df, lambda df: (df['start_timestamp'] <= df['half_date']))
        df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
            columns={'size': 'session_cnt_halfterm'})

        # merge df to result
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    # M41,Total # clicks between deadlines of each two subsequent assessments
    def assignment_total_visit_between_assignment_count(self, start, end):
        df = self.__count(self.clicks,
                          lambda df: df['start_timestamp'] >= start and df['start_timestamp'] <= end).reset_index(
            name='assignment_total_visit_between_assignment_count')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])
        # return df

    # M43, Total days active
    def active_day_cnt(self):
        df = self.dt.copy()
        df.loc[:, 'start_timestamp'] = df['start_timestamp'].dt.date

        df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['start_timestamp'].nunique().rename(
            columns={
                'start_timestamp': 'active_day_cnt'})

        # merge df to result
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    # M44, Cumulative days active for weeks 0-5 for a 10 week course
    def active_day_cnt_halfterm(self):
        df = self.session.merge(self.dt[[student_id, course_code, 'half_date']], how='inner',
                                on=[student_id, course_code])
        df = self.__loc(df, lambda df: ((df['start_timestamp'] <= df['half_date'])))
        df = self.__count(df).reset_index(name='active_day_cnt_halfterm')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M45, Cumulative days active for a 10 week course
    def active_day_cnt_total(self):
        # dictionary where key: student and course, value: list of start timestamps (just the date -> time NOT included)
        times = self.session.merge(self.dt[[student_id, course_code, 'start_week_num', 'end_week_num']], how='inner',
                                   on=[student_id, course_code])
        times['week_num'] = times['start_timestamp'].dt.week

        cut_offs = times.groupby([student_id, course_code])["start_week_num", "end_week_num"].apply(
            lambda x: x.values.tolist()).to_dict()
        times['active_day_cnt_total'] = times.groupby([student_id, course_code])['week_num'].apply(
            lambda x: pd.cut(x, list(range(cut_offs[x.name][0][0] - 1, cut_offs[x.name][0][1] + 3))))

        df = times.groupby([student_id, course_code])['active_day_cnt_total'].nunique()
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M46, Show feedback on the previous attempt
    '''
    141	== Click a specific "Attempt" in "Attempt History" in quiz
    '''

    def quiz_submission_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'] == 141).reset_index('quiz_submission_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M47, Submit a quiz attempt to be assessed
    '''
    136	== Submit Quiz
    138	== Submit Quiz
    139	== Submit Quiz (api)
    '''

    def quiz_submission_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([136, 138, 139])).reset_index(
            'quiz_submission_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M48, View front page of a quiz
    '''
    74	== Click on a module item that is a quiz
    127	== Clicked on an quiz
    128	== Clicked on an quiz (api)
    129	== click a quiz (api)
    '''

    def quiz_view_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([74, 127, 128, 129])).reset_index(
            'quiz_view_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M49, Launch a new attempt in a quiz
    '''
    130	== Start Quiz
    131	== start quiz
    132	== Start Quiz  
    '''

    def quiz_attempt_cnt(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([130, 131, 132])).reset_index(
            'quiz_attempt_cnt')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    # M50, View front page of another course
    '''
    0 == Click into a course (api)
    5 == click a course (api)

    '''

    def course_visit_count(self):
        df = self.__count(self.clicks, lambda df: df['action_code'].isin([0, 5])).reset_index('course_visit_count')
        self.result = self.result.merge(df, how='outer', on=[student_id, course_code])

        # return df

    '''
    # of clicks -> sum of event count
    '''
    def click_count(self):
        df = self.dt.groupby(["canvas_course_id","canvas_user_id"], as_index=False)["action_count"].sum().rename(columns={
            'action_count': 'tot_click_cnt'})
        #merge df to result
        self.result = self.result.merge(df,how='left', on=["canvas_course_id","canvas_user_id"])


    '''
        avg click per session
        '''
    def avg_clicks(self):
        self.result["avg_click_cnt_per_session"] = self.result['tot_click_cnt'] / self.result['session_cnt']


    '''
        avg time spent per day -> should be in min
        '''

    def avg_day_duration(self):
        df = self.dt.groupby(["canvas_course_id", "canvas_user_id", "Total_days"], as_index=False)['time_spent'].sum()
        df["avg_time_per_day"] = (df['time_spent'] / 60) / df['Total_days']  # duration is seconds -> convert to mins
        self.result = self.result.merge(df[["canvas_course_id", "canvas_user_id", "avg_time_per_day"]], how='left',
                                        on=["canvas_course_id",
                                            "canvas_user_id"])

    '''
        time (in minutes) from start term to first activity
        '''

    def start_term_to_first_activity(self):
        df = self.dt.groupby(["canvas_course_id", "canvas_user_id", "start_date"], as_index=False).agg(
            {'start_timestamp': np.min})
        # gets the min start time of start_timestamp
        df["gap_until_first_action"] = (pd.to_datetime(df.start_timestamp, utc=True,
                                                     infer_datetime_format=True).dt.tz_convert(
            'US/Eastern') - df.start_date) / np.timedelta64(1, 'm')

        df = df[["canvas_course_id", "canvas_user_id", "gap_until_first_action"]]
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    '''
        total number of participated weeks
        '''

    def total_participated_weeks(self):
        # dictionary where key: term, course, canvas, value: list of start timestamps (just the date -> time NOT included)
        times = self.dt.copy()
        times['week'] = times['start_timestamp'].dt.week

        df = times.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['week'].nunique().rename(
            columns={'week': 'tot_act_wk_cnt'})
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    '''
        irregularity of study session length (standard deviation of session length)
        '''

    def irregularity_study_session_length(self):
        df = self.dt.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['time_spent'].agg(np.std,
                                                                                                       ddof=1).rename(
            columns={'time_spent': 'session_duration_std'})
        df['session_duration_std'] = df['session_duration_std'].div(60)  # convert seconds to minutes

        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    '''
        irregularity of study interval (SD of time laspes between study sessions; time was in seconds)
        '''

    def irregularity_of_study_interval(self):
        times = self.dt[["canvas_course_id", "canvas_user_id", "start_timestamp", "end_timestamp"]]
        # turns column difference into seconds -> convert to minutes
        times['time_diff'] = (times['end_timestamp'] - times['start_timestamp']).dt.total_seconds()
        times['time_diff'] = times['time_diff'].div(60)  # convert seconds to minutes

        df = times.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)["time_diff"].agg(np.std,
                                                                                                    ddof=1).rename(
            columns={'time_diff': 'session_gap_std'})
        self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

    '''
        cumulative days active until the end of each quarter of course
        '''

    def cumulative_active_days_quarters(self):
        dt = self.dt[["canvas_course_id", "canvas_user_id", "start_timestamp", "start_date", "end_date", "Total_weeks"]]
        dt["second"] = dt['start_date'] + ((dt['Total_weeks'] / 4).apply(np.ceil) * 2).apply(
            lambda x: pd.Timedelta(x, unit='W'))  # second quarter date
        dt["third"] = dt['start_date'] + ((dt['Total_weeks'] / 4).apply(np.ceil) * 3).apply(
            lambda x: pd.Timedelta(x, unit='W'))  # third quarter date

        d2 = self.__loc(dt, lambda df: ((df['start_timestamp'] <= df['second'])))
        d3 = self.__loc(dt, lambda df: ((df['start_timestamp'] <= df['third'])))

        d1 = self.d_qrt1.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
            columns={'size': "cum_act_day_cnt_qrt1"})
        d2 = d2.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
            columns={'size': "cum_act_day_cnt_qrt2"})
        d3 = d3.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
            columns={'size': "cum_act_day_cnt_qrt3"})

        self.result = self.result.merge(d1, how='left', on=["canvas_course_id", "canvas_user_id"])
        self.result = self.result.merge(d2, how='left', on=["canvas_course_id", "canvas_user_id"])
        self.result = self.result.merge(d3, how='left', on=["canvas_course_id", "canvas_user_id"])

#-----------------------------------------------------------------------------------------------------------------------#
        '''
        Total # of clicks in the first quarter of course period (event count is # of clicks)
        '''

        def click_count_quarter(self):
            df = self.d_qrt1
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)["action_count"].sum().rename(
                columns={
                    'action_count': 'tot_click_cnt_qrt1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Average # of clicks per session in the first quarter of course period
        '''

        def avg_clicks_quarter(self):
            df = self.d_qrt1

            ds = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'tot_session_cnt_qrt1'})  #
            # num of sessions for each term,course & canvas id
            dc = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)["action_count"].sum().rename(
                columns={
                    'action_count': 'tot_click_cnt_qrt1'})
            df = ds.merge(dc, how='left', on=["canvas_course_id", "canvas_user_id"])

            df['avg_click_cnt_per_session_qrt1'] = df.tot_click_cnt_qrt1 / df.tot_session_cnt_qrt1
            self.result = self.result.merge(
                df[["canvas_course_id", "canvas_user_id", "avg_click_cnt_per_session_qrt1"]], how='left',
                on=["canvas_course_id", "canvas_user_id"])

        '''
        Total time spent (min) in the first quarter of course period
        '''

        def total_time_quarter(self):
            df = self.d_qrt1

            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['time_spent'].sum().rename(
                columns={'time_spent': 'tot_time_qrt1'})
            df['tot_time_qrt1'] = df.tot_time_qrt1.div(60)  # convert seconds to minutes

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Average session duration (min) in the first quarter of course period
        '''

        def avg_session_duration_quarter(self):
            df = self.d_qrt1

            ds = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'tot_session_cnt_qrt1'})  # num of sessions for each term,course & canvas id
            d_time = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['time_spent'].sum().rename(
                columns={'time_spent': 'tot_time_qrt1'})
            df = ds.merge(d_time, how='left', on=["canvas_course_id", "canvas_user_id"])

            df['avg_session_len_qrt1'] = (df.tot_time_qrt1.div(60) / df.tot_session_cnt_qrt1)  # converts seconds to min
            self.result = self.result.merge(df[["canvas_course_id", "canvas_user_id", "avg_session_len_qrt1"]],
                                            how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Average time spent per day (min) in the first quarter of course period
        '''

        def avg_day_duration_quarter(self):
            df = self.d_qrt1

            df = df.groupby(["canvas_course_id", "canvas_user_id", "Total_days"], as_index=False)[
                'time_spent'].sum().rename(columns={'time_spent': 'tot_time_qrt1'})
            df["avg_time_per_day_qrt1"] = (
                        df.tot_time_qrt1.div(60) / df.Total_days)  # duration is seconds -> convert to mins
            self.result = self.result.merge(df[["canvas_course_id", "canvas_user_id", "avg_time_per_day_qrt1"]],
                                            how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Length of longest inactivity (min) in the first quarter of course period
        '''

        def largest_inactivity_quarter(self):
            df = self.d_qrt1

            # start and end dates
            term = df[["canvas_course_id", "canvas_user_id", "start_date", "end_date"]].drop_duplicates()

            first = df.groupby(["canvas_course_id", "canvas_user_id"])[["start_timestamp", "end_timestamp"]].agg(
                {'start_timestamp': np.min, 'end_timestamp': np.max})
            first = first.merge(term, how='left', on=["canvas_course_id", "canvas_user_id"])
            first['start_inac'] = first['start_timestamp'] - first[
                'start_date']  # gets the inactivity between first start timestamp and start_date
            first['end_inac'] = first['end_date'] - first[
                'end_timestamp']  # gets the inactivity between last end timestamp andd end_date

            # sorts the dataframe groupby ("canvas_course_id","canvas_user_id"), ascending order of start_timestamps
            times = df.groupby(["canvas_course_id", "canvas_user_id"])[["start_timestamp", "end_timestamp"]].apply(
                lambda x: x.sort_values(['start_timestamp'])).reset_index()
            times['inactivity'] = times['start_timestamp'] - times['end_timestamp'].shift(1)

            # merges the inactivity between first inactivity and last inactvity
            df = pd.concat([times[["canvas_course_id", "canvas_user_id", 'inactivity']],
                            first[["canvas_course_id", "canvas_user_id", "start_inac"]].rename(
                                columns={'start_inac': 'inactivity'})], ignore_index=True)
            df = pd.concat([df, first[["canvas_course_id", "canvas_user_id", "end_inac"]].rename(
                columns={'end_inac': 'inactivity'})], ignore_index=True)

            # gets the longest inactvity and converts to minutes
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['inactivity'].max()
            df['inactivity'] = df['inactivity'].dt.total_seconds() / 60
            df = df.rename(columns={'inactivity': 'longest_inact_qrt1'})
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Irregularity of study effort (STD of study session length) (min) in the first quarter of course period
        '''

        def irregularity_study_session_length_quarter(self):
            df = self.d_qrt1

            df = df.groupby(["canvas_course_id", "canvas_user_id"])['time_spent'].agg(np.std, ddof=1).reset_index(
                name='irreg_session_len_qrt1')
            df['irreg_session_len_qrt1'] = df['irreg_session_len_qrt1'].div(60)  # convert seconds to minutes

            self.result = self.result.merge(df[["canvas_course_id", "canvas_user_id", "irreg_session_len_qrt1"]],
                                            how='left', on=["canvas_course_id", "canvas_user_id"])

        '''
        Irregularity of study interval (STD of time lapse between consecutive study sessions) (min) in the first quarter of course period
        '''

        def irregularity_of_study_interval_quarter(self):
            df = self.d_qrt1

            times = df[["canvas_course_id", "canvas_user_id", "start_timestamp", "end_timestamp"]]
            # turns column difference into seconds -> convert to minutes
            times['time_diff'] = (times['end_timestamp'] - times['start_timestamp']).dt.total_seconds()
            times['time_diff'] = times['time_diff'].div(60)  # convert seconds to minutes

            df = times.groupby(["canvas_course_id", "canvas_user_id"])["time_diff"].agg(np.std, ddof=1).reset_index(
                name='irreg_session_gap_qrt1')
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])


        # -----------------------------  measures from Assignment table BELOW ----------------------------------

        # Number of assignments submitted
        def assign_sub_cnt(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6am-6pm
        def assign_sub_cnt_day(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].dt.hour.between(6, 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_day'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6am-6pm
        def assign_sub_cnt_day(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & df['submitted_at'].dt.hour.between(6, 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_day'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6pm-midnight
        def assign_sub_cnt_pm(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour > 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_pm'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 12am-6am
        def assign_sub_cnt_am(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour < 6)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_am'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of on-time assignment submissions
        def on_time_assign_cnt(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'] <= df['assignment_override_due_at'].combine_first(df['assignment_due_at'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'on_time_assign_cnt'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of on-time assignment submissions
        def late_assign_cnt(self):
            df = self.assignments
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'] > df['assignment_override_due_at'].combine_first(df['assignment_due_at'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'late_assign_cnt'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        ############ Early-stage version (first quarter) ##################

        # Number of assignments submitted
        def assign_sub_cnt_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6am-6pm
        def assign_sub_cnt_day_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].dt.hour.between(6, 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_day_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6am-6pm
        def assign_sub_cnt_day_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & df['submitted_at'].dt.hour.between(6, 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_day_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 6pm-midnight
        def assign_sub_cnt_pm_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour > 18)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_pm_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of assignment submission 12am-6am
        def assign_sub_cnt_am_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'].notnull() & (df['submitted_at'].dt.hour < 6)]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'assign_sub_cnt_am_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of on-time assignment submissions
        def on_time_assign_cnt_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'] <= df['assignment_override_due_at'].combine_first(df['assignment_due_at'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'on_time_assign_cnt_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # Number of late assignment submissions
        def late_assign_cnt_qtr1(self):
            df = self.d_assgn_qrt1
            df = df[df['submission_state'].isin(['graded', 'submitted', 'pending_review'])]
            df = df[df['submitted_at'] > df['assignment_override_due_at'].combine_first(df['assignment_due_at'])]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'late_assign_cnt_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # -----------------------------  measures from Assignment table ABOVE -----------------------------------------------------------------------------------------------------

        # -----------------------------  measures from Discussion table BELOW -----------------------------------------------------------------------------------------------------

        # total number of discussion forum POSTS grouped by student, course, term
        def disc_post_count(self):
            df = self.discussion[self.discussion['parent_discussion_post_id'] == -1]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_post_cnt'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # total number of discussion forum posts REPLIES grouped by student, course, term
        def disc_reply_count(self):
            # discussion table
            df = self.discussion[self.discussion['parent_discussion_post_id'] != -1]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_reply_cnt'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # total number of messages posted (includes post & reply)
        def disc_tot(self):
            df = self.discussion
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_tot_messages'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg word count per post (discussion forum)
        def avg_wordcount_per_post(self):
            df = self.discussion[self.discussion['parent_discussion_post_id'] == -1]

            # calculates the avg count per post -> sum(messages)/# of posts
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_post'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg word count per reply (discussion forum)
        def avg_wordcount_per_reply(self):
            df = self.discussion[self.discussion['parent_discussion_post_id'] != -1]

            # calculates the avg count per reply -> sum(messages)/# of replies
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_reply'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg # words in forum posts (post & reply)
        def avg_wordcount_tot(self):
            df = self.discussion

            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_tot'})
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg depth per post (post & reply)
        def avg_post_depth(self):
            df = self.discussion

            dcount = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_post_cnt'})
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_depth'].sum()

            df = df.merge(dcount, how='outer', on=["canvas_course_id", "canvas_user_id"])
            df['avg_depth_post'] = df.discussion_post_depth / df.disc_post_cnt

            df = df[["canvas_course_id", "canvas_user_id", 'avg_depth_post']]
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        ############# Early-stage version (first quarter) #####################

        # total number of discussion forum POSTS grouped by student, course, term
        def disc_post_count_qtr1(self):
            df = self.d_disc_qrt1[self.d_disc_qrt1['parent_discussion_post_id'] == -1]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_post_cnt_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # total number of discussion forum posts REPLIES grouped by student, course, term
        def disc_reply_count_qtr1(self):
            # discussion table
            df = self.d_disc_qrt1[self.d_disc_qrt1['parent_discussion_post_id'] != -1]
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_reply_cnt_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # total number of messages posted (includes post & reply)
        def disc_tot_qtr1(self):
            df = self.d_disc_qrt1
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_tot_messages_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg word count per post (discussion forum)
        def avg_wordcount_per_post_qtr1(self):
            df = self.d_disc_qrt1[self.d_disc_qrt1['parent_discussion_post_id'] == -1]

            # calculates the avg count per post -> sum(messages)/# of posts
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_post_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg word count per reply (discussion forum)
        def avg_wordcount_per_reply_qtr1(self):
            df = self.d_disc_qrt1[self.d_disc_qrt1['parent_discussion_post_id'] != -1]

            # calculates the avg count per reply -> sum(messages)/# of replies
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_reply_qtr1'})

            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg # words in forum posts (post & reply)
        def avg_wordcount_tot_qtr1(self):
            df = self.d_disc_qrt1

            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_content'].apply(
                lambda x: np.mean(x.str.len())).rename(columns={'discussion_post_content': 'avg_word_tot_qtr1'})
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])

        # avg depth per post (post & reply)
        def avg_post_depth_qtr1(self):
            df = self.d_disc_qrt1

            dcount = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False).size().rename(
                columns={'size': 'disc_post_cnt'})
            df = df.groupby(["canvas_course_id", "canvas_user_id"], as_index=False)['discussion_post_depth'].sum()

            df = df.merge(dcount, how='outer', on=["canvas_course_id", "canvas_user_id"])
            df['avg_depth_post_qtr1'] = df.discussion_post_depth / df.disc_post_cnt

            df = df[["canvas_course_id", "canvas_user_id", 'avg_depth_post_qtr1']]
            self.result = self.result.merge(df, how='left', on=["canvas_course_id", "canvas_user_id"])


if __name__ == "__main__":
    # mc = MeasureCalculator(files)
    # df.to_csv(os.path.join(os.getcwd(), "csvFiles", 'random_testing.csv'))

    '''
    Need to keep in mind and start thinking about batches for code
    can maybe do one instance/object per a year worth of data?
    then merge those resulting tables with each oher
    ending result table --> one table

    could possible do a measure calculator for every year? and/or very quarter
    each having a result table which we can merge together?
    '''
