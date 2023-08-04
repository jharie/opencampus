
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import pandas as pd
import random


#deftest()

#HLAC計算
def test1(i,j, image, filter):
  y,x = image.shape
  m,n = filter.shape
  p = np.power(image[i:i+m,j:j+m],filter)
  count = np.sum(p&(filter!=0)*p)
  return count

def test(i,j, image, filter):
  y,x = image.shape
  m,n = filter.shape
  count = np.sum(image[i:i+m,j:j+m]**3&(filter==3))


def whitening(img):
  # Point 1: 黒色部分に対応するマスク画像を生成
  mask = np.all(img[:,:,:] == [0, 0, 0], axis=-1)

  # Point 2: 元画像をBGR形式からBGRA形式に変換
  skelton = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

  # Point3: マスク画像をもとに、白色部分を透明化
  skelton[mask,3] = 0
  return skelton

#HLAC用の3値マスク
#2(2乗)を畳み込みの計算の便宜的に，3(3乗)を

hlac_filters =  [np.array([[False, False, False], [False,  True, False], [False, False, False]]),  np.array([[False, False, False], [False,  True,  True], [False, False, False]]),  np.array([[False, False,  True], [False,  True, False], [False, False, False]]),  np.array([[False,  True, False], [False,  True, False], [False, False, False]]),  np.array([[ True, False, False], [False,  True, False], [False, False, False]]),  np.array([[False, False, False], [ True,  True,  True], [False, False, False]]),  np.array([[False, False,  True], [False,  True, False], [ True, False, False]]),  np.array([[False,  True, False], [False,  True, False], [False,  True, False]]),  np.array([[ True, False, False], [False,  True, False], [False, False,  True]]),  np.array([[False, False,  True], [ True,  True, False], [False, False, False]]),  np.array([[False,  True, False], [False,  True, False], [ True, False, False]]),  np.array([[ True, False, False], [False,  True, False], [False,  True, False]]),  np.array([[False, False, False], [ True,  True, False], [False, False,  True]]),  np.array([[False, False, False], [False,  True,  True], [ True, False, False]]),  np.array([[False, False,  True], [False,  True, False], [False,  True, False]]),  np.array([[False,  True, False], [False,  True, False], [False, False,  True]]),  np.array([[ True, False, False], [False,  True,  True], [False, False, False]]),  np.array([[False,  True, False], [ True,  True, False], [False, False, False]]),  np.array([[ True, False, False], [False,  True, False], [ True, False, False]]),  np.array([[False, False, False], [ True,  True, False], [False,  True, False]]),  np.array([[False, False, False], [False,  True, False], [ True, False,  True]]),  np.array([[False, False, False], [False,  True,  True], [False,  True, False]]),  np.array([[False, False,  True], [False,  True, False], [False, False,  True]]),  np.array([[False,  True, False], [False,  True,  True], [False, False, False]]),  np.array([[ True, False,  True], [False,  True, False], [False, False, False]])]


def mult_conv(image, filter):
  #feature_map = signal.convolve2d(image, filter, mode='valid')
  #count = np.sum(feature_map)
  y,x = image.shape
  n,m = filter.shape
  y= y-n+1
  x= x-n+1
  count = list(map(lambda j: list(map(lambda i: test(i,j, image,filter), range(y))),range(x)))
  return np.sum(count)

# パッチ版HLAC特徴量
def split_into_batches(image, nx, ny):
    batches = []
    for y_batches in np.array_split(image, ny, axis=0):
        for x_batches in np.array_split(y_batches, nx, axis=1):
            batches.append(x_batches)
    return batches

def extract_batchwise_hlac_3(image, hlac_filters, nx, ny):
    batches = split_into_batches(np.uint8(image), nx, ny)
    hlac_filters = np.uint8(hlac_filters)
    hlac_batches = []
    with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as e:
        for batch in batches:
            result = list(e.map(mult_conv, [batch] * len(hlac_filters), hlac_filters))
            hlac_batches.append(result)
    return np.array(hlac_batches)

def extract_batchwise_hlac(image, hlac_filters, nx, ny):
    batches = split_into_batches(np.uint8(image), nx, ny)
    hlac_filters = np.uint8(hlac_filters)
    hlac_batches = []
    extracter = lambda args: np.sum(signal.convolve2d(args[0], args[1], mode='valid') == np.sum(args[1]))
    with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as e:
        for batch in batches:
            result = list(e.map(extracter, zip([batch] * len(hlac_filters), hlac_filters)))
            hlac_batches.append(result)
    return np.array(hlac_batches)


def vector_angle(hv1, hv2, eps = 1e-6):
    hv1 = (hv1 + eps) / np.linalg.norm(hv1 + eps)
    hv2 = (hv2 + eps) / np.linalg.norm(hv2 + eps)
    return np.arccos(np.clip(np.dot(hv1, hv2), -1.0, 1.0))

def visualize(image, name, hlac_angles, nx, ny, color, thikness, th=0.1):
    batches = split_into_batches(image, nx, ny)
    dst = np.zeros_like(image)
    hlac_angles -= np.nanmin(hlac_angles)
    hlac_angles /= np.nanmax(hlac_angles)
    py = 0
    count = 0
    for y in range(ny):
        px = 0
        for x in range(nx):
            batch = batches[y * nx + x]
            angle = hlac_angles[y * nx + x]
            if angle > th:
              #第一引数：画像，第二引数：長方形の左上頂点，第三：右下頂点，第四：線の色，第五：thikness(線の太さ)
              dummy = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), color, thikness)
              dst = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]),  color, thikness)
              count +=1
            px += batch.shape[1]
        py += batch.shape[0]
    if count == 0:
      return(image)
    cv2.imwrite(name+"dst_hlac.png", dst)
    wht1 = whitening(dst)
    cv2.imwrite(name+"wht.png", wht1)
    x1, y1, x2, y2 = 0, 0, wht1.shape[1], wht1.shape[0]
    # 合成!
    frame1 = image
    frame1[y1:y2, x1:x2] = frame1[y1:y2, x1:x2] * (1 - wht1[:, :, 3:] / 255) + \
                      wht1[:, :, :3] * (wht1[:, :, 3:] / 255)

    cv2.imwrite(name+"frame_hlac1.png", frame1)
    #return cv2.addWeighted(dummy, 0.2, image, 0.8, 0)
    return frame1

def visualize1(image, name, hlac_angles, nx, ny, th=0.3, side_cut=False, up_down_cut = False):
    batches = split_into_batches(image, nx, ny)
    dst = np.zeros_like(image)
    dummy = np.zeros_like(image)
    hlac_angles -= np.nanmin(hlac_angles)
    hlac_angles /= np.nanmax(hlac_angles)

    py = 0

    for y in range(ny):
        px = 0
        for x in range(nx):
          batch = batches[y * nx + x]
          angle = hlac_angles[y * nx + x]
          if angle > th :
            if side_cut and x==0 or (side_cut and x == nx-1):
              print('cut')
            elif up_down_cut and y==0 or (up_down_cut and y == ny-1):
              print('cut')
            else:
              #print(x)
              #第一引数：画像，第二引数：長方形の左上頂点，第三：右下頂点，第四：線の色，第五：thikness(線の太さ)
              #dst = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, int(255 * angle), 0), -1)
              #rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, int(255 * angle), int(255 * angle)), 1)
              dst = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, 255, 0), 2)
              #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
              dummy = cv2.rectangle(dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, 255, 0), 2)
              #dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
          px += batch.shape[1]
        py += batch.shape[0]

    cv2.imwrite(name+"dst_hlac.png", dst)
    wht1 = whitening(dst)
    cv2.imwrite(name+"wht.png", wht1)
    x1, y1, x2, y2 = 0, 0, wht1.shape[1], wht1.shape[0]
    # 合成!
    frame1 = image
    frame1[y1:y2, x1:x2] = frame1[y1:y2, x1:x2] * (1 - wht1[:, :, 3:] / 255) + \
                      wht1[:, :, :3] * (wht1[:, :, 3:] / 255)

    return frame1
    cv2.imwrite(name+"frame_hlac1.png", frame1)
    #mask = frame.astype(bool)
    #return cv2.addWeighted(dst, 0.4, cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), 0.6, 0)
    #return cv2.addWeighted(dummy, 0.2, image, 0.8, 0)

def InfoScore(data,thresh,bins=100):
    num_hist, range_hist = np.histogram(data, bins= bins)
    mean_hist = (range_hist[1:] + range_hist[:-1]) / 2
    p = num_hist/num_hist.sum()
    N = p.shape[0]
    M = np.sum(mean_hist>thresh)
    if (M== 0):
        return np.inf
    q = np.zeros(N)
    q[mean_hist>thresh] = 1 / M
    Dqp = - np.log(M)  - np.sum(q*np.log(p))
    return Dqp

def main():
  c_list = ["Red", "Blue", "Yellow", "Black", "White"]
  RGB =[(255,69,0)[::-1], (135,206,250)[::-1], (255,244,244)[::-1], (0,0,0)[::-1], (255,244,244)[::-1]]
  #牧場の画像，犬の画像
  image_list = [ ('./drive/MyDrive/ref1.png', './drive/MyDrive/tar1.png') ,('./drive/MyDrive/5ref1.png','./drive/MyDrive/5tar1.png'),('./drive/MyDrive/cat1.png','./drive/MyDrive/cat2.png')]
  #image_list = glob.glob("./drive/MyDrive/*")
  if 'flag' not in st.session_state:
    st.session_state["flag"] = False
  if 'color' not in st.session_state:
    st.session_state["color"] = RGB[0]#選択しないときの初期化(赤)
  if 'image' not in st.session_state:
    st.session_state['image'] = image_list[0]

  nx=int(st.sidebar.number_input('横の分割サイズ：',2,30,20))
  ny=int(st.sidebar.number_input('縦の分割サイズ：',2,30,20))
  selected_color = st.sidebar.selectbox(
    '分割ブロックの線の色を選択：',
  c_list
  )

  st.session_state["color"] =RGB[ c_list.index(selected_color)]

  line=int(st.sidebar.number_input('分割ブロックの線の太さ：',1,3,1))
  # リロードボタン
  if st.sidebar.button('画像切り替え'):
    st.session_state["flag"] = False
    st.session_state['image'] = random.choice(image_list)

  print(st.session_state['image'][0])
  reference = cv2.imread(st.session_state['image'][0])
  target = cv2.imread(st.session_state['image'][1])

  image_name = 'test1'

  col1, col2 = st.columns(2)
  with col1:
    #height = reference.shape[0]
    #width = reference.shape[1]
    #reference = cv2.resize(reference , (int(width*2), int(height*2)))
    st.header("左の画像")
    st.image(reference, use_column_width=True, channels='BGR')
  with col2:
    #height = target.shape[0]
    #width = target.shape[1]
    #target = cv2.resize(target , (int(width*2), int(height*2)))
    st.header("右の画像")
    st.image(target, use_column_width=True, channels='BGR')

  if st.button("間違い箇所の表示"):
    #3値化
    #img1_gray_bin = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    #img2_gray_bin = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    #2値化
    img1_gray_bin = cv2.threshold(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 255
    img2_gray_bin = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 255


    #nx, ny = 15, 15
    reference_hlac = extract_batchwise_hlac(img1_gray_bin, hlac_filters, nx, ny)
    target_hlac = extract_batchwise_hlac(img2_gray_bin, hlac_filters, nx, ny)

    st.session_state["r_hlac"] = reference_hlac
    st.session_state["t_hlac"] = target_hlac

    hlac_angles = [vector_angle(rv, tv) for rv, tv in zip(reference_hlac, target_hlac)]
    hlac_angles_n = hlac_angles
    hlac_angles_n -= np.nanmin(hlac_angles)
    hlac_angles_n /= np.nanmax(hlac_angles)

    out = visualize(target,image_name, hlac_angles, nx, ny, st.session_state["color"], line)#5引数:精度の閾値，6引数：横分割の場合の分割線付近の誤差を無視，7引数：縦分割の場合の分割線付近の誤差を無視
    st.image(out, channels='BGR')
    cv2.imwrite(image_name+"out.png", out)
    st.session_state["flag"] = True

  #解析ボタンの表示
  if st.button('解析'):
    print(st.session_state["flag"])
    if st.session_state["flag"]:
      #arr = np.random.normal(1, 1, size=100)
      #fig, ax = plt.subplots()
      #ax.hist(arr, bins=20)
      #st.pyplot(fig)

      # 可視化
      fig = plt.figure()
      ax = fig.add_subplot(1,3,1)
      ax.set_title('Left')
      plt.imshow( st.session_state["r_hlac"], aspect='auto', cmap='gray')

      ax = fig.add_subplot(1,3,2)
      ax.set_title('Right')
      plt.imshow( st.session_state["t_hlac"], aspect='auto', cmap='gray')

      ax = fig.add_subplot(1,3,3)
      ax.set_title('Difference')
      plt.imshow( st.session_state["t_hlac"]- st.session_state["r_hlac"], aspect='auto', cmap='gray')
      fig.tight_layout()
      st.pyplot(fig)

if __name__ == '__main__':
  main()
