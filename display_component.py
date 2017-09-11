
import smt_process.detect_class as detect_class
import sys

MODEL="/home/junwon/smt-project/SMT/detect_part/cd_20170714.model"

DC  = detect_class.ComponentDetector()

def progressbar_callback(msg, i, m):
    """
    Callback to display a text based progress bar when loading,
     storing or performing any other lengthy operation.

    :param msg: Message to be displayed
    :param i: Current value
    :param m: Maximal value
    :return:
    """
    print msg,
    x = int(100 * float(i) / m)
    print "#" * (x / 2),
    print "[%2d%%]" % x,
    print "-" * (50 - x / 2),
    print "\r",
    sys.stdout.flush()


DC.get_components_from_pickle(MODEL,callback=progressbar_callback)



