import numpy as np
from pylab import *
from scipy.stats import ttest_rel
import pickle

row_order = [
  ('sci.med', 13),
  ('sci.space', 14),
  ('sci.crypt', 11),
  ('sci.electronics', 12),
  ('comp.graphics', 1),
  ('comp.os.ms-windows.misc', 2),
  ('comp.sys.ibm.pc.hardware', 3),
  ('comp.sys.mac.hardware', 4),
  ('comp.windows.x', 5),
  ('misc.forsale', 6),
  ('rec.autos', 7),
  ('rec.motorcycles', 8),
  ('rec.sport.baseball', 9),
  ('rec.sport.hockey', 10),
  ('talk.politics.guns', 16),
  ('talk.politics.mideast', 17),
  ('talk.politics.misc', 18),
  ('alt.atheism', 0),
  ('soc.religion.christian', 15),
  ('talk.religion.misc', 19)
]

col_order = [
  ('medical', 13),
  ('space', 14),
  ('cryptography', 11),
  ('electronics', 12),
  ('graphics', 1),
  ('windows', 2),
  ('pc', 3),
  ('mac', 4),
  ('x11', 5),
  ('sell', 6),
  ('car', 7),
  ('motorcycle', 8),
  ('baseball', 9),
  ('hockey', 10),
  ('gun', 16),
  ('middle east', 17),
  ('politics', 18),
  ('atheism', 0),
  ('christianity', 15),
  ('religion', 19)
]

figure(1)
ary = pickle.load(open('centrality.pickle'))
row_order = row_order[::-1]
sense_diffs = []
arranged = np.zeros((20, 20))
for row in xrange(20):
    for col in xrange(20):
        arranged[row,col] = ary[row_order[row][1], col_order[col][1]]
        if row != col: sense_diffs.append(arranged[row,row]-arranged[row,col])
yticks(arange(20)+0.5, [ro[0] for ro in row_order])
xticks(arange(20)+0.5, [co[0] for co in col_order], rotation='vertical')
pcolor(arranged, cmap='hot')
title('Topic alignment with common sense')
subplots_adjust(bottom=0.22, left=0.3)
clim(-60, 10)
cbar = colorbar(ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
cbar.set_label('Centrality')

figure(2)
ary = pickle.load(open('nosense.centrality.pickle'))

arranged = np.zeros((20, 20))
no_sense_diffs = []
for row in xrange(20):
    for col in xrange(20):
        arranged[row,col] = ary[row_order[row][1], col_order[col][1]]
        if row != col: no_sense_diffs.append(arranged[row,row]-arranged[row,col])
yticks(arange(20)+0.5, [ro[0] for ro in row_order])
xticks(arange(20)+0.5, [co[0] for co in col_order], rotation='vertical')
pcolor(arranged, cmap='hot')
title('Topic alignment without common sense')
subplots_adjust(bottom=0.22, left=0.3)
clim(-60, 10)
cbar = colorbar(ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
cbar.set_label('Centrality')

print np.mean(sense_diffs), np.mean(no_sense_diffs)
print ttest_rel(no_sense_diffs, sense_diffs)
show()

