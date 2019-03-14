#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'Comment:Demonstration how to define and use function,class and model.'

__author__ = 'wilsonwong' #作者

def test():
    print('Track:demo_model.py=>test()');
    import model1.demo_model
    import model2.demo_model
    model1.demo_model.test();
    model2.demo_model.test();
    model1.model11.demo_model.test();
    model1.model12.demo_model.test();
    model2.model21.demo_model.test();
    model2.model22.demo_model.test();

if __name__=='__main__':
    print('Track:demo_model.py=>__main__');
    test();
