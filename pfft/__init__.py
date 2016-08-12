from .version import __version__

from .core import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
