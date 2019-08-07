import sys
import PIL
import os

import numpy as np

from keras.preprocessing.image import load_img


def jitter_bbox(img_path, bbox, mode, ratio):
	"""
	This method jitters the position or dimensions of the bounding box.
	:param img_path: The to the image
	:param bbox: The bounding box to be jittered
	:param mode: The mode of jittere:
	'same' returns the bounding box unchanged
		  'enlarge' increases the size of bounding box based on the given ratio.
		  'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
		  'move' moves the center of the bounding box in each direction based on the given ratio
		  'random_move' moves the center of the bounding box in each direction by randomly
						sampling a value in [-ratio,ratio)
	:param ratio: The ratio of change relative to the size of the bounding box.
		   For modes 'enlarge' and 'random_enlarge'
		   the absolute value is considered.
	:return: Jittered bounding box
	"""

	assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
			'mode %s is invalid.' % mode

	if mode == 'same':
		return bbox

	img = load_img(img_path)

	if mode in ['random_enlarge', 'enlarge']:
		jitter_ratio  = abs(ratio)
	else:
		jitter_ratio  = ratio

	if mode == 'random_enlarge':
		jitter_ratio = np.random.random_sample()*jitter_ratio
	elif mode == 'random_move':
		# for ratio between (-jitter_ratio, jitter_ratio)
		# for sampling the formula is [a,b), b > a,
		# random_sample * (b-a) + a
		jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

	jit_boxes = []
	for b in bbox:
		bbox_width = b[2] - b[0]
		bbox_height = b[3] - b[1]

		width_change = bbox_width * jitter_ratio
		height_change = bbox_height * jitter_ratio

		if width_change < height_change:
			height_change = width_change
		else:
			width_change = height_change

		if mode in ['enlarge','random_enlarge']:
			b[0] = b[0] - width_change //2
			b[1] = b[1] - height_change //2
		else:
			b[0] = b[0] + width_change //2
			b[1] = b[1] + height_change //2

		b[2] = b[2] + width_change //2
		b[3] = b[3] + height_change //2

		# Checks to make sure the bbox is not exiting the image boundaries
		b = bbox_sanity_check(img.size, b)
		jit_boxes.append(b)
	# elif crop_opts['mode'] == 'border_only':
	return jit_boxes


def squarify(bbox, squarify_ratio, img_width):
	"""
	Changes is the ratio of bounding boxes to a fixed ratio
	:param bbox: Bounding box
	:param squarify_ratio: Ratio to be changed to
	:param img_width: Image width
	:return: Squarified boduning box
	"""
	width = abs(bbox[0] - bbox[2])
	height = abs(bbox[1] - bbox[3])
	width_change = height * squarify_ratio - width
	bbox[0] = bbox[0] - width_change/2
	bbox[2] = bbox[2] + width_change/2
	# Squarify is applied to bounding boxes in Matlab coordinate starting from 1
	if bbox[0] < 0:
		bbox[0] = 0

	# check whether the new bounding box goes beyond image boarders
	# If this is the case, the bounding box is shifted back
	if bbox[2] > img_width:
		# bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
		bbox[0] = bbox[0]-bbox[2] + img_width
		bbox[2] = img_width
	return bbox


def update_progress(progress):
	"""
	Shows the progress
	:param progress: Progress thus far
	"""
	barLength = 20 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)

	block = int(round(barLength*progress))
	text = "\r[{}] {:0.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()


def img_pad(img, mode = 'warp', size = 224):
	"""
	Pads a image given the boundries of the box needed
	:param img: The image to be coropped and/or padded
	:param mode: The type of padding or resizing:
			warp: crops the bounding box and resize to the output size
			same: only crops the image
			pad_same: maintains the original size of the cropped box  and pads with zeros
			pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
						the desired output size in that direction while maintaining the aspect ratio. The rest
						of the image is	padded with zeros
			pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
					in which case it scales the image down, and then pads it
	:param size: Target size of image
	:return:
	"""
	assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
	image = img.copy()
	if mode == 'warp':
		warped_image = image.resize((size,size), PIL.Image.NEAREST)
		return warped_image
	elif mode == 'same':
		return image
	elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
		img_size = image.size  # size is in (width, height)
		ratio = float(size)/max(img_size)
		if mode == 'pad_resize' or	\
			(mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
			img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
			image = image.resize(img_size, PIL.Image.NEAREST)
		padded_image = PIL.Image.new("RGB", (size, size))
		padded_image.paste(image, ((size-img_size [0])//2,
					(size-img_size [1])//2))
		return padded_image


def bbox_sanity_check(img_size, bbox):
	"""
	Confirms that the bounding boxes are within image boundaries.
	If this is not the case, modifications is applied.
	:param img_size: The size of the image
	:param bbox: The bounding box coordinates
	:return: The modified/original bbox
	"""
	img_width, img_heigth = img_size
	if bbox[0] < 0:
		bbox[0] = 0.0
	if bbox[1] < 0:
		bbox[1] = 0.0
	if bbox[2] >= img_width:
		bbox[2] = img_width - 1
	if bbox[3] >= img_heigth:
		bbox[3] = img_heigth - 1
	return bbox


def get_path(file_name='',
			 save_folder='models',
			 dataset='pie',
			 save_root_folder='data/'):
	"""
	A path generator method for saving model and config data. It create directories if needed.
	:param file_name: The actual save file name , e.g. 'model.h5'
	:param save_folder: The name of folder containing the saved files
	:param dataset: The name of the dataset used
	:param save_root_folder: The root folder
	:return: The full path for the model name and the path to the final folder
	"""
	save_path = os.path.join(save_root_folder, dataset, save_folder)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	return os.path.join(save_path, file_name), save_path

