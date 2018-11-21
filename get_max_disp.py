item=[line.split()[-1] for line in open('lib/datasets/data/cityscapes/annotations/train.txt', 'r').read().splitlines()]
from PIL import Image
import numpy as np
max_d = -1
for it in item:
    print ('deal with %s' % it)
    disp = np.asarray(Image.open('lib/datasets/data/cityscapes/'+it), dtype=np.float32)
    disp /= 255
    max_d = np.max([max_d, np.max(disp)])
#max disp is 126.49803924560547
print (max_d)
