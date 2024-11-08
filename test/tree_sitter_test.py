import sys
import json


sys.path.append(r'./')

from codesecurity.feature.objects import CommonFeatureSet

test_file='dataset/gcj_cpp/0x03BB/3264486_5633382285312000_0x03BB.cpp'


feature=CommonFeatureSet.from_file(test_file)


print(feature.tokens)