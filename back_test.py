# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-22
Description: 
"""
from inference import SlotModel
from datetime import datetime

texts = [
		'灯调到最亮', '调暗客厅灯', '把客厅灯开亮', '空调调到2档', '风扇调大一点', '风扇调小', '声音调大点', '声音小一点', '把灯调成红色', '晾衣架升起来', '晾衣架降下来',
		'晾衣架调高2米', '晾衣架下降2米', '打开浴霸的照明', '关闭浴霸的照明', '扫地机回去充电', '扫地机在哪',
		'打开客厅浴霸的照明', '给我开空气净化器UV杀菌', '帮我开启阳台的晾衣架的UV杀菌', '开投影仪', '打开投影仪', '我要开开主卧的扫地机器人',
		'我要开开主卧浴霸的开关', '我要开开主卧浴霸', '开开主卧浴霸', '加湿器请开启', '俺2天后想要让服务员预约房间打扫的服务', '可不可以请让前台222222222星期之后帮我送一点东西',
		'主卧灯调亮', '把主卧灯调亮一点', '把灯调亮', '电视机大声一点', '把灯调亮50', '开客厅窗帘',
		'打开客厅空调', '打开客厅电视', '打开厨房空调', '打开卧室取暖器', '打开客厅的空调', '打开客厅的电视', '打开厨房的空调', '打开卧室的取暖器',
		'打开香薰机的灯', '洗衣机', '电视机', '客厅', '空调', '打开客厅空调', '晾衣架升高20%', '晾衣架的高度调高2米', '跟彩云城又尤为患者', '六月十日',
		'将客厅的空调调高2度', '我要看湖南卫视', '设置空调为制热模式', '红色', '10秒', '到10秒', '调高一点', 'go go go', '晾衣机升高20%', '晾衣杆调高点',
		'让晾衣杆调高些', '帮我让晾衣机向上抬升些', '晾衣架升高2米', '把温度调高2', '温度调高2度', '湿度调高2', '温度降低3度', '湿度降低3', '把卧室灯设为调至50',
		'把空调调到25度', '调到1档', '把灯调到100%', '风扇调成1档', '空调调成23度', '雾量调到1档', '把客厅灯的颜色调为白色', '音量调到20%', '加湿器调到1档',
		'晾衣架升高2米', '打开浴霸的照明', '帮我关浴霸的UV杀菌', '呼叫客房清洁服务', '查一下厨房的消毒柜干燥模式打开了吗', '开启儿童房传感器UV杀菌模式', '5天后给我预约一下维修服务',
		'请让酒店给我送555双订书器', '空调调节到30', '我想看新闻联播', '帮我查下空调的照明开着吗', '帮我查下香薰机的照明开关开着没']

intents = [
		'set_brightness', 'decrement_brightness', 'increment_brightness', 'set_wind_speed', 'increment_wind_speed',
		'decrement_wind_speed', 'increment_attribute', 'decrement_attribute', 'set_color', 'set_max_height', 'set_min_height',
		'increment_height', 'decrement_height', 'open_zhaoming', 'close_zhaoming', 'power_charge', 'open_seek',
		'open_zhaoming', 'open_function', 'open_function', 'turn_on', 'turn_on', 'turn_on', 'turn_on', 'turn_on',
		'turn_on', 'turn_on', 'room_clean', 'room_supplement', 'increment_brightness', 'increment_brightness',
		'increment_brightness', 'increment_volume', 'increment_brightness', 'turn_on',
		'turn_on', 'turn_on', 'turn_on', 'turn_on', 'turn_on', 'turn_on', 'turn_on', 'turn_on',
		'turn_on', None, None, None, None, 'turn_on', 'increment_height', 'increment_attribute',
		None, None, 'increment_attribute', 'select_channel', 'set_attribute', None, None, None,
		'increment_attribute', None, 'increment_height', 'increment_height', 'increment_height', 'increment_height',
		'increment_height', 'increment_attribute', 'increment_attribute', 'increment_attribute', 'decrement_attribute',
		'decrement_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute',
		'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'increment_height',
		'open_zhaoming', 'close_function', 'room_clean', 'query_attribute', 'open_function', 'room_maintain',
		'room_supplement', 'set_attribute', 'query_videos', 'query_zhaoming', 'query_zhaoming']

# 生词识别
texts = [
		'婴儿辅食机调到迷糊模式', '把婴儿辅食机设置到肉泥模式', '把婴儿辅食机设置到鱼泥模式', '把婴儿辅食机设置到果泥模式', '把婴儿辅食机设置到煮沸模式',
		'把婴儿辅食机设置到蒸煮模式', '把婴儿辅食机设置到轻度搅拌模式', '把婴儿辅食机设置到中度搅拌模式', '把婴儿辅食机设置到重度搅拌模式', '打开紫外线消毒柜',
		'关闭紫外线消毒柜', '把紫外线消毒柜设置到自动模式', '把紫外线消毒柜设置到消毒模式', '把紫外线消毒柜设置到烘干模式', '把紫外线消毒柜设置到酸奶模式',
		'把紫外线消毒柜设置到果干模式', '把紫外线消毒柜设置到消毒毛绒玩具模式', '把紫外线消毒柜设置到消毒积木模式', '把紫外线消毒柜设置到消毒毛巾模式',
		'把紫外线消毒柜设置到消毒餐具模式', '把紫外线消毒柜设置到消毒衣物模式', '把紫外线消毒柜设置到消毒电子产品模式']
intents = [
		'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute',
		'set_attribute', 'set_attribute', 'turn_on', 'turn_off', 'set_attribute', 'set_attribute', 'set_attribute',
		'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute',
		'set_attribute']

slot_model = SlotModel.read_configs(1)
for t, it in zip(texts, intents):
	start_time = datetime.now()
	res = slot_model.inference(t, it, False, None)
	print('text: %s, intent: %s res: %s, time costs: %s' % (t, it, res, (datetime.now() - start_time).total_seconds()))
