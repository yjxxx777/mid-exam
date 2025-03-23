輸出結果:https://youtu.be/OdTClJyhIqg
# 動機：

  1. 車道線偵測是 自動駕駛、智慧交通 及 輔助駕駛系統（ADAS） 的重要技術之一。

  2. 手機架設於儀表板或行車記錄器中，可即時監測車道，輔助駕駛。

# 目的：

  1. 透過 影像處理與邊緣檢測技術，偵測行車畫面中的車道線，並提供視覺化的標記。

  2. 強化 自動導航 及 駕駛輔助功能，提升行車安全。

  3. 優化處理效能，使其適用於影片串流，而不僅僅是靜態影像。
  
# 方法：

### 影像處理與邊緣偵測

  1. 轉灰階（Grayscale）：減少計算量，使特徵更明顯。

  2. 高斯模糊（Gaussian Blur）：去除雜訊，使邊緣更平滑。

  3. Canny 邊緣偵測（Canny Edge Detection）：強化邊緣，突出車道線。

### 感興趣區域（ROI）

  1. 過濾掉不必要的部分（如天空、遠景），專注於路面。

  2. 設定一個梯形的 ROI 來框住車道範圍。

### 霍夫變換（Hough Transform）

  1. 用 cv2.HoughLinesP() 偵測直線，並篩選出可能的車道線。

  2. 分析線段的 斜率，以區分 左側車道線 和 右側車道線。

### 平滑處理

  1. 使用 移動平均法 平滑偵測出的車道線，使其更穩定。

### 優化並適用於影片

  1. 使用 OpenCV 讀取影片，並對每一幀影像做相同的處理。

  2. 透過 滑動視窗技術，追蹤車道變化，減少雜訊影響。

# 優化點：
1. 支援影片輸入：從 cv2.VideoCapture() 讀取影片並逐幀處理。

2. ROI 過濾雜訊：設定 梯形區域 只關注車道線，提升準確率。

3. 調整霍夫變換參數： minLineLength=30 過濾掉過短的線段，maxLineGap=200 避免過多碎片化線段。

4. 綠色標記車道線，清楚標示辨識結果。

5. 即時顯示結果，可用 q 鍵退出。

