# -*- coding: utf-8 -*-
dict_temp = {2:1,3:5,1:10}
a = sorted(dict_temp.items(),key=lambda x:x[1])
print(a)
