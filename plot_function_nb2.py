# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 08:59:07 2021

@author: diego
"""

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()