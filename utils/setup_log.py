import logging
import datetime
import os.path as osp
import os
import subprocess
from termcolor import colored
# pip install gitpython
import git

'''
class DuplicateFilter:
    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, ext_tb):
        self.logger.removeFilter(self)
'''

class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()
        self.msgs_times = {}

    def filter(self, record):
        rv = record.msg not in self.msgs
        if rv:
            self.msgs.add(record.msg)
            self.msgs_times[record.msg] = 1
        else:
            if self.msgs_times[record.msg] < 3:
                self.msgs_times[record.msg] += 1
                rv = True
            else:
                pass
        return rv

    def filter_once(self, record):
        rv = record.msg not in self.msgs
        if rv:
            self.msgs.add(record.msg)
        return rv

    def filter_all(self, record):
        return True


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def git_info():
    # https://gitpython.readthedocs.io/en/stable/reference.html#module-git.objects.commit
    # TestingTools/model_performance_evaluation_tool/src/common/job_runner.py
    #     subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
    repo_path = osp.join(osp.dirname(__file__), '../')
    repo = git.Repo(repo_path)
    return repo, repo.head.commit.hexsha, repo.is_dirty()


def setup_log(log_name):
    # Initialize logging
    # simple_format = '%(levelname)s >>> %(message)s'
    medium_format = (
        '%(levelname)s : %(filename)s[%(lineno)d]'
        ' >>> %(message)s'
    )

    # Reference:
    #   http://59.125.118.185:8088/ALG/TestingTools/-/blob/master/model_performance_evaluation_tool/src/common/testingtools_log.py
    formatter = logging.Formatter(
                '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

    medium_format_new = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    dt_now = datetime.datetime.now()
    minute_ex = (dt_now.minute // 10) * 10
    dt_text = f'{dt_now.year:04d}{dt_now.month:02d}{dt_now.day:02d}_{dt_now.hour:02d}-{minute_ex:02d}'
    name_tag = osp.splitext(log_name)
    new_log_name = f'{name_tag[0]}_{dt_text}{name_tag[1]}'

    new_log_name = log_name

    get_log_file = osp.join(osp.dirname(__file__), f'../logs/{new_log_name}')
    if not osp.exists(osp.dirname(get_log_file)):
        os.makedirs(osp.dirname(get_log_file))

    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format_new
    )

    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print('@{} created at {}'.format(get_log_file, dt_now))

    repo, hexsha, is_dirty = git_info()
    logging.info(f'  {type(repo)}')
    logging.info(f'  repo.head.commit.hexsha: {hexsha}')
    logging.info(f'  is_dirty():              {is_dirty}')

    print(f'  {type(repo)}')
    print(f'  repo.head.commit.hexsha: {hexsha}')
    print(f'  is_dirty():              {is_dirty}')

    if is_dirty and False:
        git_diff = subprocess.check_output(['git', 'diff'])
        logging.info(f'  git diff\n{git_diff.decode()}')
        print(f'  git diff\n{git_diff.decode()}')

    print('\n===== log_file: {}\n'.format(get_log_file))


trace_logger = logging.getLogger(__name__)
dup_filter = DuplicateFilter()
trace_logger.addFilter(dup_filter)
