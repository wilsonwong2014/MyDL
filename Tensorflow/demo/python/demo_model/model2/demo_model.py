#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'Comment:demo_model/model2/demo_model.py.'

__author__ = 'wilsonwong' #作者

def test():
    print('Track:demo_model/model2/demo_model.py=>test()');
    import model21.demo_model
    import model22.demo_model
    model21.demo_model.test();
    model22.demo_model.test();

if __name__=='__main__':
    print('Track:demo_model/model2/demo_model.py.=>__main__');
    test();
