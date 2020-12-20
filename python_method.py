# staticmethod
@staticmethod
can be used w/o initialize the class 

@staticmethod
# specify the datatype, return type
def create_datetime(time:str) -> datetime.datetime:
    pass


# logging
import logging
# hasattr() returns true if an object ha the given named attribute and false if it does not.
if hasattr(sys,"frozen"):
    program_dir = os.path.split(os.path.realpath(sys.executable))[0]
else:
    program_dir = os.path.split(sys.argv[0])[0]

from argparse import ArgumentParser
# exit program after execution
if __name__ = "__main__":
    arguments=[]
    sys.exit(main(arguments))
def main(argv=None) ->int:
    if argv:
        logger.info(f"appending main arguments {argv} to sys args")
        sys.argv.extend(argv[0])
    parser = ArgumentParser(
            description=program_licence + program_usage + time_format, formatter_class=RawDescriptionHelpFormatter
            )
    try:
        # setup argument parser
        parser.add_argument(
                "-s",
                "--start",
                dest="start",
                type=str,
                action="store",
                default=None,
                help='starting date the time in "YYYY-MM-DD HH:mm:ss" \n[default: "' + hour_ago + '"]',
                metavar="TIME",
                }
        parse.add_argument("-V","--version",action="version",version=program_version_message)
        parse.add_argument("-l", "--log", dest="log",action="count",default=0, help="""Archives the logs analyzed""")

        # Process arguments
        args = parser.parse_args()
        start = args.start
        log = args.log

        #
        readiness_test = AnalyzerTest()

    expect KeyboardInterrupt:
        logger.info("Keyboard Interrupt Encountered")
        logger.info("Exiting")
        return 0 # successful exit
    except Exception as e:
        logging.info(F"Ran into unhandled exception {e}")
        
        if DEBUG or TESTRUN or verbose:
            raise e:
        # print help normally
        parser.printusage()
        print(program_usage)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + " for help use -help\n")
        # 
        if log:
            logger.info("archiving logs")
            readiness_test.archive_logs(prog_dir)
        return 2    # unix command line systan error



# Unittest
import unittest
import calc

class TestCAlc(unittest.TestCase):
    def test_add(self): # must start with test_
        result = calc.add(10,5)
        self.assertEqual(result,15)

        # context manager
        with self.assertRaise(ValueError):
            calc.divie(10,0)

if __name__ == '__main__':
    unitest.main()

# Class
# class variable : shared by all class
class Employee:
    raise_amount = 1.04 # class variable
    def __init__(self,first,pay):
        self.first = first
        self.pay = pay
    def apply_raise(self):
        self.pay = int(self.pay * Employee.raise_amount)
        # or
        self.pay = int(self.pay * self.raise_amount)

emp_1 = Employee('first',4000)
Employee.raise_amount = 1.06
print(Employee.raise_amount)    #->1.06
print(emp_1.raise_amount)   # ->1.04

print(emp_1.__dict__) # print name space of emp_1
print(isinstance(mgr1,Manager))
print(issubclass(Manager,Employee))

