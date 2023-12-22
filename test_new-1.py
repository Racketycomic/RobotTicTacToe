import cv2
import sys
import time
import numpy as np
from tito import findBestMove,evaluate
from pymycobot.mycobot import MyCobot
import serial
cap_num =2
cap = cv2.VideoCapture(cap_num)


class Tic():
    def __init__(self, camera_x = 160, camera_y = 15):
        self.color = 0
        # parameters to calculate camera clipping parameters 计算相机裁剪参数的参数
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        # set cache of real coord 设置真实坐标的缓存
        self.cache_x = self.cache_y = 0
        self.camera_x, self.camera_y = camera_x, camera_y
        # The coordinates of the cube relative to the mycobot280
        # 立方体相对于 mycobot 的坐标
        self.c_x, self.c_y = 0, 0
        # The ratio of pixels to actual values
        # 像素与实际值的比值
        self.ratio = 0
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Get ArUco marker params. 获取 ArUco 标记参数
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        self.HSV = {
            # "yellow": [np.array([11, 85, 70]), np.array([59, 255, 245])],
            #"yellow": [np.array([22, 93, 0]), np.array([45, 255, 245])],
            "red": [np.array([220, 70, 100]), np.array([255, 160, 150])],
            "green": [np.array([35, 43, 35]), np.array([90, 255, 255])], #green
            "blue": [np.array([100, 110, 200]), np.array([124, 255, 255])], #blue
            # "blue":[np.array([100, 43, 46]), np.array([124, 255, 255])]
            # "cyan": [np.array([78, 43, 46]), np.array([99, 255, 255])],
        }
        self.bounding_box_color = {"red":(0,0,255),"green":(0,255,0),"blue":(255,0,0)}
        self.width = None
        self.height = None
        self.game_board = np.zeros([3,3])
        self.mc = None
        self.move_coords = [
            [132.2, -136.9, 200.8, -178.24, -3.72, -107.17],  # D Sorting area
            [238.8, -124.1, 204.3, -169.69, -5.52, -96.52], # C Sorting area
            [115.8, 177.3, 210.6, 178.06, -0.92, -6.11], # A Sorting area
            [-6.9, 173.2, 201.5, 179.93, 0.63, 33.83], # B Sorting area
        ]
        self.move_angles = [
            [0.61, 45.87, -92.37, -41.3, 2.02, 9.58],  # init the point
            [18.8, -7.91, -54.49, -23.02, -0.79, -14.76],  # point to grab
        ]
        self.waypoint = [132.2, -136.9, 240.8, -178.24, -3.72, -107.17]

        self.state_dict = {'0':0,'1':0,'2':0}
        
        # get real serial
        self.plist = [
            str(x).split(" - ")[0].strip() for x in serial.tools.list_ports.comports()
                
        ]
    
    def resizeFrame(self,frame):
        # enlarge the image by 1.5 times
            fx = 1.5
            fy = 1.5
            
            frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy,
                            interpolation=cv2.INTER_CUBIC)
            if self.x1 != self.x2:
                # the cutting ratio here is adjusted according to the actual situation
                frame = frame[int(self.y2*0.78):int(self.y1*1.1),
                              int(self.x1*0.86):int(self.x2*1.08)]
            return frame
    def get_calculate_params(self, img):
        # Convert the image to a gray image 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect ArUco marker.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        """
        Two Arucos must be present in the picture and in the same order.
        There are two Arucos in the Corners, and each aruco contains the pixels of its four corners.
        Determine the center of the aruco by the four corners of the aruco.
        """
        if len(corners) > 0:
            if ids is not None:
                if len(corners) <= 1 or ids[0] == 1:
                    return None
                x1 = x2 = y1 = y2 = 0
                point_11, point_21, point_31, point_41 = corners[0][0]
                x1, y1 = int((point_11[0] + point_21[0] + point_31[0] + point_41[0]) / 4.0), int(
                    (point_11[1] + point_21[1] + point_31[1] + point_41[1]) / 4.0)
                point_1, point_2, point_3, point_4 = corners[1][0]
                x2, y2 = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4.0), int(
                    (point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4.0)
                
                return x1, x2, y1, y2
        return None

    # set camera clipping parameters 设置相机裁剪参数 
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    # set parameters to calculate the coords between cube and mycobot280
    # 设置参数以计算立方体和 mycobot 之间的坐标
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0/ratio

    def draw_marker(self, img, x, y):
        # draw rectangle on img 在 img 上绘制矩形
        cv2.rectangle( 
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # add text on rectangle
        cv2.putText(img, "({},{})".format(x, y), (x, y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (243, 0, 0), 2,)

    
    def color_detect(self, img):
        # set the arrangement of color'HSV
        result_array = {}
        # img = self.draw_tic_tac_toe_board(img)
        for mycolor, item in self.HSV.items():
            result_array[mycolor] = []
            x = y = 0
            color = mycolor
            # print("mycolor:",mycolor)
            redLower = np.array(item[0])
            redUpper = np.array(item[1])

            # transfrom the img to model of gray 将图像转换为灰度模型
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # print("hsv",hsv)

            # wipe off all color expect color in range 擦掉所有颜色期望范围内的颜色
            mask = cv2.inRange(hsv, item[0], item[1])

            # a etching operation on a picture to remove edge roughness
            # 对图片进行蚀刻操作以去除边缘粗糙度
            erosion = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=2)

            # the image for expansion operation, its role is to deepen the color depth in the picture
            # 用于扩展操作的图像，其作用是加深图片中的颜色深度
            dilation = cv2.dilate(erosion, np.ones(
                (1, 1), np.uint8), iterations=2)

            # adds pixels to the image 向图像添加像素
            target = cv2.bitwise_and(img, img, mask=dilation)

            # the filtered image is transformed into a binary image and placed in binary
            # 将过滤后的图像转换为二值图像并放入二值
            ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)

            # get the contour coordinates of the image, where contours is the coordinate value, here only the contour is detected
            # 获取图像的轮廓坐标，其中contours为坐标值，这里只检测轮廓
            contours, hierarchy = cv2.findContours(
                dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # do something about misidentification
                boxes = [
                    box
                    for box in [cv2.boundingRect(c) for c in contours]
                    if min(img.shape[0], img.shape[1]) / 10
                    < min(box[2], box[3])
                    < min(img.shape[0], img.shape[1]) / 1
                ]
                if boxes:
                    for box in boxes:
                        x, y, w, h = box

                        if abs(x) + abs(y) > 0:
                            result_array[mycolor].append((x,y))
                        # find the largest object that fits the requirements 找到符get_position合要求的最大对象
                        c = max(contours, key=cv2.contourArea)
                        # get the lower left and upper right points of the positioning object
                        # 获取定位对象的左下和右上点
                        x, y, w, h = cv2.boundingRect(c)
                        # locate the target by drawing rectangle 通过绘制矩形来定位目标
                        cv2.rectangle(img, (x, y), (x+w, y+h), (153, 153, 0), 2)
                        # calculate the rectangle center 计算矩形中心
                        x, y = (x*2+w)/2, (y*2+h)/2
                        # calculate the real coordinates of mycobot280 relative to the target
                        #  计算 mycobot 相对于目标的真实坐标
                        for pic, contour in enumerate(contours): 
                                area = cv2.contourArea(contour) 
                                if(area > 300): 
                                    x, y, w, h = cv2.boundingRect(contour) 
                                    img = cv2.rectangle(img, (x, y), 
                                                            (x + w, y + h), 
                                                            self.bounding_box_color[mycolor], 2) 
                                    
                                    cv2.putText(img, mycolor, (x, y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                                                self.bounding_box_color[mycolor])
                        
        return result_array,frame
    
    def draw_tic_tac_toe_board(self,image):
        height, width, _ = image.shape
        self.width = width
        self.height = height

        # Define the number of rows and columns in the Tic Tac Toe board
        rows = 3
        cols = 3

        # Calculate the spacing between lines based on the image dimensions
        horizontal_spacing = width // (cols)
        vertical_spacing = height // (rows )
        ranges = {'x':[0],'y':[0]}
        # Draw horizontal lines
        for i in range(1, rows):
            y = i * vertical_spacing
            image = cv2.line(image, (0, y), (width, y), (0, 0, 0), 2)
            ranges['y'].append(y)



        # Draw vertical lines
        for j in range(1, cols):
            x = j * horizontal_spacing
            image =cv2.line(image, (x, 0), (x, height), (0, 0, 0), 2)
            ranges['x'].append(x)

        ranges['x'].append(width)
        ranges['y'].append(height)

            
        # print(ranges)
        return image
    
    def detect_cube_in_grid(self,cube_x, cube_y):
        cell_width = self.width // 3
        cell_height = self.height // 3

        for row in range(3):
            for col in range(3):
                cell_x_start, cell_x_end = col * cell_width, (col + 1) * cell_width
                cell_y_start, cell_y_end = row * cell_height, (row + 1) * cell_height

                if cell_x_start <= cube_x < cell_x_end and cell_y_start <= cube_y < cell_y_end:
                    return row,col
    
    def update_board(self,result_array):
        for keys,value in result_array.items():
            for i in value:
                cubex,cubey = i
                row,col = self.detect_cube_in_grid(cubex,cubey)
                if self.game_board[row][col] !=0:
                    if keys == 'green':
                        self.game_board[row][col] = 1
                    else:
                        self.game_board[row][col] = 2
                else:
                    print("Position is taken")
    
    def find_turn(self):
        a, counts = np.unique(self.game_board,return_counts=True)
        for i in range(3):
            for j in range(3):
                if self.game_board[i][j] == 0:
                    self.state_dict['0']+=1
                elif self.game_board[i][j] == 1:
                    self.state_dict['1']+=1
                elif self.game_board[i][j] == 2:
                    self.state_dict['2']+=1
        if self.state_dict['0']!=9 and self.state_dict['1'] ==  self.state_dict['2']:
            return True
        else:
            return False
        
    def get_grid_pos(self,row, col):
        cell_width = self.width // 3
        cell_height = self.height // 3

        center_x = (col + 0.5) * cell_width
        center_y = (row + 0.5) * cell_height

        return int(center_x), int(center_y)
    
    def run(self):
        self.mc = MyCobot(self.plist[0], 115200) 
        self.mc.send_angles([0.61, 45.87, -92.37, -41.3, 2.02, 9.58], 20)
        time.sleep(2.5)
    
    def move(self,x,y):
        self.mc.send_coords(self.waypoint,40)
        time.sleep(3)
        self.pump_on()
        time.sleep(2)
        self.mc.send_coords(self.move_coords[0],40)
        time.sleep(8)
        self.mc.send_coords([x, y, 150, 179.87, -3.78, -62.75], 25, 0)
        time.sleep(3)

        self.mc.send_coords([x, y, 103, 179.87, -3.78, -62.75], 40, 1)
        time.sleep(3)
        self.pump_off()
        self.mc.send_coords([x, y, 133, 179.87, -3.78, -62.75], 40, 1)
        time.sleep(3)
        self.mc.send_angles(self.move_angles[0], 25)
        

    def pump_on(self):
        # 让2号位工作
        self.mc.set_basic_output(2, 0)
        # 让5号位工作
        self.mc.set_basic_output(5, 0)

    # 停止吸泵 m5
    def pump_off(self):
        # 让2号位停止工作
        self.mc.set_basic_output(2, 1)
        # 让5号位停止工作
        self.mc.set_basic_output(5, 1)

    def get_position(self, x, y):
        return ((y - self.c_y)*self.ratio + self.camera_x), ((x - self.c_x)*self.ratio + self.camera_y)



_init_ = 20  
init_num = 0
nparams = 0
num = 0
real_sx = real_sy = 0
detect = Tic()
detect.run()
while cv2.waitKey(1) < 0:
    _,frame = cap.read()
    # frame = cv2.imread('aikit_V2/AiKit_280M5/scripts/captured_image_0(2).jpg')
    if _init_ > 0:
            _init_ -= 1
            continue
    frame = detect.resizeFrame(frame)
    if init_num < 20:
            if detect.get_calculate_params(frame) is None:
                cv2.imshow("figure", frame)
                continue
            else:
                x1, x2, y1, y2 = detect.get_calculate_params(frame)
                detect.draw_marker(frame, x1, y1)
                detect.draw_marker(frame, x2, y2)
                detect.sum_x1 += x1
                detect.sum_x2 += x2
                detect.sum_y1 += y1
                detect.sum_y2 += y2
                init_num += 1
                continue
    elif init_num == 20:
        detect.set_cut_params(
            (detect.sum_x1)/20.0,
            (detect.sum_y1)/20.0,
            (detect.sum_x2)/20.0,
            (detect.sum_y2)/20.0,
        )
        detect.sum_x1 = detect.sum_x2 = detect.sum_y1 = detect.sum_y2 = 0
        init_num += 1
        continue
    
        # print()

    
    # calculate params of the coords between cube and mycobot280 计算立方体和 mycobot 之间坐标的参数
    if nparams < 10:
        if detect.get_calculate_params(frame) is None:
            cv2.imshow("figure", frame)
            continue
        else:
            x1, x2, y1, y2 = detect.get_calculate_params(frame)
            detect.draw_marker(frame, x1, y1)
            detect.draw_marker(frame, x2, y2)
            detect.sum_x1 += x1
            detect.sum_x2 += x2
            detect.sum_y1 += y1
            detect.sum_y2 += y2
            nparams += 1
            continue
    elif nparams == 10:
        nparams += 1
        # calculate and set params of calculating real coord between cube and mycobot280
        # 计算和设置计算立方体和mycobot之间真实坐标的参数
        detect.set_params(
            (detect.sum_x1+detect.sum_x2)/20.0,
            (detect.sum_y1+detect.sum_y2)/20.0,
            abs(detect.sum_x1-detect.sum_x2)/10.0 +
            abs(detect.sum_y1-detect.sum_y2)/10.0
        )
        print("ok")
        continue
    frame = detect.draw_tic_tac_toe_board(frame)
    result_array,frame = detect.color_detect(frame)
    print(result_array)
    cv2.imshow('figure',frame)
    input("")
    detect.update_board(result_array)
    # print(detect.game_board)

    # if not detect.find_turn():
    if move[0] ==  -10:
        print("Its a draw")
        sys.exit()
    else:
        move = findBestMove(detect.game_board)
        print(detect.game_board)
        print("move",move)
        grid_x,grid_y = detect.get_grid_pos(move[0],move[1])
        real_x,real_y = detect.get_position(grid_x,grid_y)
        detect.move(real_x,real_y)
        winner = evaluate(detect.game_board)
        if winner != 0 :
            print("The winner is ",winner)
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        else:
            continue

  