# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:09:49 2017

@author: 牛帥
"""

import numpy as np
import cv2
import screeninfo
#import matplotlib.pyplot as plt


def f(a):
    p = np.array([[0, 0, 0, a[0], a[1], 1, -a[3]*a[0], -a[3]*a[1], -a[3]],
                  [a[0], a[1], 1, 0, 0, 0, -a[2]*a[0], -a[2]*a[1], -a[2]]])          #行列を表す
    return p


def estimate_homography(points):                 #ホモグラフィ推定関数
    p =[]                                        #点を展開して、Aは8＊8の行列、Bは余った行列
    for a in points:
        p.append(f(a))

    P = np.vstack(p)
    A = P[:, :8]
    B = P[:, 8:]

    invA = np.linalg.inv(A)
    x = np.dot(invA, -B)
    x = invA.dot(-B)                       #Ph = 0  Ax+b = 0  Ax = -b
    h = np.vstack((x,1))                   #x = (-b)/A
    H = h.reshape(3,3)                     #Hは３＊３を変更するg
    return H


def perspective_transform(H, point):        #Hを利用して、プロジェクタ座標を計算する
    point = np.hstack((point, 1)).reshape(3, 1)
    point = H.dot(point)
    point = point / point[2]
    point = point.reshape(3)
    return point[:2]


def readCamera():
    cap = cv2.VideoCapture(0) #Videoを利用する
    cap.grab()
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("frame.png", frame)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    screen_id = 1
    # get the size of the screeng
    screen = screeninfo.get_monitors()[screen_id]         #モニタsーの解像度を獲得する
    width, height = screen.width, screen.height

    p_width = 854
    p_height = 480

    #自動座標調整、プロジェクタ実験の部分
    cw_pairs = np.loadtxt("camera_worldpoints.txt") #カメラ、ワールド座標
    H_wc = estimate_homography(cw_pairs)
    H_cw = np.linalg.inv(H_wc)         #カメラ、ワールドのホモグラフィを求めた

#    #pc_pairs = np.loadtxt("pro_cpoints.txt")
#    H_cp = estimate_homography(pc_pairs)
#    H_pc = np.linalg.inv(H_cp)

    readCamera()                        #以下はマーカー点の検出
    image = cv2.imread("frame.png")
    ret, corners = cv2.findChessboardCorners(image, (10, 7))
    cv2.drawChessboardCorners(image,(10, 7),corners,ret)
    #cv2.imshow("corners",image)

    points_c = np.array([corners[0],
                         corners[9],
                         corners[60],
                         corners[69]])
    points_c = points_c.reshape(-1, 2)
    points_w = []
    for point_c in points_c:
        point_w = perspective_transform(H_wc, point_c)
        points_w.append(point_w)
    points_w = np.vstack(points_w)

#    points_p = []
#    for point_c in points_c:
#        point_p = perspective_transform(H_pc, point_c)
#        points_p.append(point_p)
#    points_p = np.vstack(points_p)
    points_p = np.loadtxt("point_p.txt")
    pw_pairs = np.hstack((points_p, points_w))
    H_wp = estimate_homography(pw_pairs)
    H_pw = np.linalg.inv(H_wp)

    points_worldRect = np.loadtxt("targets.txt")   #ワールド長方形座標
    points_projectorRect = []
    for point_worldRect in points_worldRect:
        point_projectorRect = perspective_transform(H_pw, point_worldRect)
        points_projectorRect.append(point_projectorRect)

    points_projectorRect = np.vstack(points_projectorRect)
    points_s = np.array([[0, 0],
                         [p_width, 0],
                         [p_width, p_height],      #スクリーン座標系を作る(点の座標)じゅ
                         [0, p_height]])
    sp_pairs = np.hstack((points_s, points_projectorRect))
    H_ps = estimate_homography(sp_pairs)
    H_sp = np.linalg.inv(H_ps)


    dst = cv2.warpPerspective(image, H_ps, (p_width, p_height))
    image = cv2.resize(image, (p_width, p_height))
    dst = cv2.warpPerspective(image, H_ps, (p_width, p_height))
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x, screen.y)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('flower_s.png',dst)