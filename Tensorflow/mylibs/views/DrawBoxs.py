#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
#在一张图像上绘制矩形框序列
def DrawBoxs(img,boxs,texts=None,colors=None):
    fig, ax = plt.subplots()
    plt.imshow(img)
    for i,box in enumerate(boxs):
        x=box[0]
        y=box[1]
        w=box[2]-box[0]
        h=box[3]-box[1]
        text='' if texts is None else texts[i]
        color='red' if colors is None else colors[i]
        ax.add_patch(patches.Rectangle((x,y),w,h,fill=False,lw=1,color=color))
        plt.text(x, y, text, fontdict={'size': 16, 'color': color})
    #ax.set_xlim(x_min, x_max)
    #ax.set_ylim(y_min, y_max)
    plt.show()

#============================
if __name__=='__main__':
    boxs=np.array(
        [
            [10,10,50,80],
            [40,40,90,70],        
        ]
        )
    titles=['a','b']
    img=np.zeros((100,200,3),dtype=np.uint8)
    DrawBoxs(img,boxs,texts=titles,colors=['red','green'])


