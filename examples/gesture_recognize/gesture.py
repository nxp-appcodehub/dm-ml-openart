# Rock Paper Scissors Game
### Copyright 2023 NXP  ##

import sensor, image, time, machine, pyb, os, tf,gc
import lvgl as lv
import lvgl_helper
import micropython
import math
from imagetools import get_png_info, open_png

sensor.reset()
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)
sensor.set_auto_whitebal(True)
sensor.set_brightness(2)
sensor.set_contrast(1)
sensor.set_gainceiling(16)
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.set_windowing((160, 160))       # Set 240x240 window.
sensor.set_framerate(0<<9 | 1<<13)
sensor.skip_frames(time = 2000)          # Wait for settings take effect.
sensor.set_auto_gain(True)

clock = time.clock()                # Create a clock object to track the FPS.



#Initialize LVGL
lv.init()
#lvgl task hander called in timer
def timer_callback(self):
    lv.tick_inc(10)
    lv.task_handler()
    pyb.mdelay(5)

timer = machine.Timer(1)
timer.init(50)
timer.callback(timer_callback)
#initialize display interface
lv_disp_buf = lv.disp_buf_t()
LVGL_W,LVGL_H = lvgl_helper.get_display_size()
if LVGL_W > 640 and LGVL_H > 480:
    LVGL_W = 640 #reduce the display res size lvgl背景大小
    LVGL_H = 480
buf1_1 = lvgl_helper.alloc(LVGL_W, LVGL_H)
buf1_2 = lvgl_helper.alloc(LVGL_W, LVGL_H)
lv_disp_buf.init(buf1_1, buf1_2, LVGL_W*LVGL_H)
disp_drv = lv.disp_drv_t()
disp_drv.init()
disp_drv.buffer = lv_disp_buf
disp_drv.flush_cb = lvgl_helper.flush

disp_drv.hor_res = LVGL_W
disp_drv.ver_res = LVGL_H
disp_drv.register()

indev_drv = lv.indev_drv_t()
indev_drv.init()
indev_drv.type = lv.INDEV_TYPE.POINTER
indev_drv.read_cb = lvgl_helper.capture

# declare the screen to show
scr = lv.obj()
#declare image to show picture from camera
img = lv.img(lv.scr_act())
img.align(lv.scr_act(), lv.ALIGN.CENTER, 0, 0)
#declare a label to show the result
label = lv.label(lv.scr_act())
label.set_pos(220,20)
label.set_text("From AI:")
style = lv.style_t()

#show cursor of touch panel
tp = indev_drv.register()
cursor=lv.img(lv.scr_act())
cursor.set_src(lv.SYMBOL.GPS)
tp.set_cursor(cursor)
# declare the screen to show
scr = lv.obj()

def LV_DPX(n):
    if n == 0:
        n = 0
    else:
        dpi = lv.disp_t.get_dpi(None)
        if int((dpi*n+80)/160) > 1 :
            n = int((dpi*n+80)/160)
        else:
            n = 1
    return n
def LV_MATH_MAX(m,n):
    if m>n:
        return m
    else :
        return n
def LV_MATH_MIN(m,n):
    if m>n:
        return n
    else :
        return m

LV_DPI = 100

ui_screen_image_camera = lv.img(lv.scr_act(), None)
style_screen_image_camera_main = lv.style_t()
style_screen_image_camera_main.init()
style_screen_image_camera_main.set_image_recolor(lv.STATE.DEFAULT, lv.color_make(0xff, 0xff, 0xff))
style_screen_image_camera_main.set_image_recolor_opa(lv.STATE.DEFAULT, 0)
style_screen_image_camera_main.set_image_opa(lv.STATE.DEFAULT, 255)
ui_screen_image_camera.add_style(ui_screen_image_camera.PART.MAIN, style_screen_image_camera_main)
ui_screen_image_camera.set_pos(10, 50)
ui_screen_image_camera.set_size(160, 160)

def show_camera_image(image,w,h):
    img_dsc = lv.img_dsc_t(
        {
            "header": {"always_zero": 0, "w": w, "h": h, "cf": lv.img.CF.TRUE_COLOR},
            "data_size": w*h*2,
            "data": lvgl_helper.get_ptr(image),
        }
    )
    ui_screen_image_camera.set_src(img_dsc)


# Register PNG image decoder
decoder = lv.img.decoder_create()
decoder.info_cb = get_png_info
decoder.open_cb = open_png
#declare png buttons

with open('/sd/gesture_s.png','rb') as f:
    img_data_gestures = f.read()
    f.close()

img_dsc_gestures = lv.img_dsc_t({
    "header": {"always_zero": 0, "w": 160, "h": 160, "cf": lv.img.CF.TRUE_COLOR},
    'data_size': len(img_data_gestures),
    'data': img_data_gestures
})

with open('/sd/gesture_r.png','rb') as f:
    img_data_gesturer = f.read()
    f.close()

img_dsc_gesturer = lv.img_dsc_t({
    "header": {"always_zero": 0, "w": 160, "h": 160, "cf": lv.img.CF.TRUE_COLOR},
    'data_size': len(img_data_gesturer),
    'data': img_data_gesturer
})

with open('/sd/gesture_p.png','rb') as f:
    img_data_gesturep = f.read()
    f.close()

img_dsc_gesturep = lv.img_dsc_t({
    "header": {"always_zero": 0, "w": 160, "h": 160, "cf": lv.img.CF.TRUE_COLOR},
    'data_size': len(img_data_gesturep),
    'data': img_data_gesturep
})

with open('/sd/welcome.png','rb') as f:
    img_data_welcome = f.read()
    f.close()

img_dsc_welcome = lv.img_dsc_t({
    "header": {"always_zero": 0, "w": 160, "h": 160, "cf": lv.img.CF.TRUE_COLOR},
    'data_size': len(img_data_welcome),
    'data': img_data_welcome
})
#图片显示位置
img_show = lv.img(lv.scr_act())
lv.img.cache_set_size(2)
img_show.set_src(img_dsc_welcome)
img_show.set_pos(180, 50)



mobilenet = "gesture.tflite"
net = tf.load(mobilenet)
clock = time.clock()
list_gesture=["Paper","Rock","Scissors"]


label2 = lv.label(lv.scr_act())
label2.set_pos(20,20)
label2.set_text('From You:')



pattern_mode = 1; # 1easy;2medium;3hard
cir = 1;
game_start = 0

def event_btn1(obj,evt):
    global pattern_mode
    global cir
    global game_start

    if evt == lv.EVENT.CLICKED:

        if game_start == 0:
            game_start = 1
            label1.set_text("Stop")
        else:
            game_start = 0
            label1.set_text("Start")


btn1 = lv.btn(lv.scr_act())
btn1.set_fit2(lv.FIT.NONE,lv.FIT.TIGHT)
btn1.set_pos(360,200)
btn1.set_size(80,40)
label1 = lv.label(btn1)
label1.set_text("Start")
btn1.set_event_cb(event_btn1)


def event_dropdown1(obj,evt):
    global pattern_mode
    if evt == lv.EVENT.VALUE_CHANGED:
        option = " "*20
        pattern.get_selected_str(option,len(option))
        print(option[0])
        if(int(option[0]) == 1):
            pattern_mode = 1
        elif(int(option[0]) == 2):
            pattern_mode = 2
        elif(int(option[0]) == 3):
            pattern_mode = 3

pattern = lv.dropdown(lv.scr_act())
pattern.set_pos(360,100)
pattern.set_width(110)
pattern.set_options("1 Easy\n2 Medium\n3 Hard")
pattern.set_event_cb(event_dropdown1)


while(1):
    img = sensor.snapshot()
    clock.tick()

    show_camera_image(img,img.width(),img.height())

    if(game_start == 1):
        img_gesture = sensor.snapshot()

        values = tf.classify(net, img_gesture)
        result = values[0].output()


        m = 0
        max_value = 0.0
        for idx,value in enumerate(result):
            if value > max_value:
                m = idx
        print(result)
        print(m)
        label2.set_text("From You: "+list_gesture[m])

        if(pattern_mode == 1): #
            img_show.set_src(img_dsc_gesturer)
        elif(pattern_mode == 2): #
            if(cir == 1):
                img_show.set_src(img_dsc_gestures)
            elif(cir == 2):
                img_show.set_src(img_dsc_gesturer)
            elif(cir == 3):
                img_show.set_src(img_dsc_gesturep)
            cir = (cir + 1) % 3
        elif(pattern_mode == 3): #
            if(m == 0):
                img_show.set_src(img_dsc_gestures)
            elif(m == 1):
                img_show.set_src(img_dsc_gesturep)
            elif(m == 2):
                img_show.set_src(img_dsc_gesturer)

lvgl_helper.free()
scr.delete()
lv.deinit()
timer.deinit()



