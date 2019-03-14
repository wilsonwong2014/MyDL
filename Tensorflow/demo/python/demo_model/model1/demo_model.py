#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'Comment:demo_model/model1/demo_model.py.'

__author__ = 'wilsonwong' #作者

def test():
    print('Track:demo_model/model1/demo_model.py=>test()');
    import model11.demo_model
    import model12.demo_model
    model11.demo_model.test();
    model12.demo_model.test();

if __name__=='__main__':
    print('Track:demo_model/model1/demo_model.py.=>__main__');
    test();
