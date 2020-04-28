# We use those imports to :
# - manipulalte the images
import numpy as np
import cv2
from base64 import b64decode, b64encode
from PIL import Image
from io import BytesIO
# - create a responsive interface within the cells
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from google.colab import output
# - hide the warnings
import warnings
# - create new directory on the drive
from google.colab import drive
import os
# - to shuffle lines
import random


#######################################################################################################################################
### METHODS TO MANIPULATE PICTURES ###
######################################

def base64_to_ndarray(data):
	'''
	Convert a base64 image into a ndarray
	'''
	arr = data.split(',')[1] # the first part is just some metadata about format
	arr = b64decode(arr) # decode the data, make it a string
	arr = np.fromstring(arr, np.uint8) # convert the image into an array (line)
	arr = cv2.imdecode(arr, cv2.IMREAD_COLOR) # convert it into the BRG 
	arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) # convert it to RGB
	return arr

def ndarray_to_base64(arr):
	'''
	Convert a ndarray image into a jpeg base64 byte object
	'''
	b64 = Image.fromarray(arr) # create a PIL image from the array
	buff = BytesIO() # a byte buffer to write the image in
	b64.save(buff, format="JPEG") # save the image into the buffer
	b64 = b64encode(buff.getvalue()) # finally encode the data
	buff.close() # close the buffer
	return b64

def rect_to_box(rect, bbox=None):
	'''
	Transform a rect into a box. Span in the bbox if needed
	'''
	box = np.array(rect).reshape(-1)
	box[2:] += box[:2]
	if bbox is not None:
		bbox = np.array(bbox).reshape(-1)
		box[:2] += bbox[:2]
		box[2:] += bbox[:2]
	return box

def box_to_rect(box):
	'''
	Convert a box into e rect
	'''
	rect = np.array(box).reshape(-1)
	rect[2:] -= rect[:2]
	return rect

def expend(box, margin, w,h):
	'''
	Expend a box by a certin margin. We cn also add bounds for the width and height
	'''
	# get the margin
	margin = np.array(margin).reshape(-1)
	l = margin.size
	m = np.zeros(4)
	if l==1:
		m = margin[0]*np.array([-1,-1,1,1])
	elif l==2:
		m = margin[0]*np.array([-1,0,1,0]) + margin[1]*np.array([0,-1,0,1])
	elif l==4:
		m = np.copy(margin)
	else:
		raise Exception("Wrong size for margin : {}".format(l))
	# apply the margin
	new_box = np.array(box) + m
	new_box[:2] = np.maximum(new_box[:2],[0,0])
	new_box[2:] = np.minimum(new_box[2:],[w,h])
	return new_box

def centered_square_box(box):
	'''
	Get the biggest square box contained and centered into a rectangular box.
	'''
	box = np.array(box).reshape(-1)
	center = (box[2:]+box[:2])/2
	size = np.min(box[2:]-box[:2])/2
	xmin = center[0]-size
	xmax = center[0]+size
	ymin = center[1]-size
	ymax = center[1]+size
	return np.array([xmin,ymin,xmax,ymax]).astype(np.uint32)

def cutter(img, box, copy=False):
	'''
	Get a part of an image. We can decide either we want a copy of it or the real thing.
	'''
	x0,y0, x1,y1 = box
	if copy:
		return np.copy(img[y0:y1,x0:x1])
	else:
		return img[y0:y1,x0:x1]

def draw_box(img, box, color=(0,0,0)):
	'''
	Draw a rectangle box on an image. Default colour is pitch black.
	'''
	x0,y0, x1,y1 = box
	return cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)

def draw_boxes(img, boxes, color):
	'''
	Draw multiples boxes on the image
	'''
	for box in boxes:
		draw_box(img, box, color)
	return


#######################################################################################################################################
### METHODS ABOUT THE TEXTFILE ###
##################################

def read_textfile(path):
	'''
	Read the content of the textfile (debug method)
	'''
	f = open(path, "r")
	lines = f.readlines()
	for line in lines:
		print(line)
	f.close()
	return

def reset_textfile(path):
	'''
	/!| Warning /!|
	This method will delete the content of the textfile (debug method)
	'''
	f = open(path, "w")
	f.close()
	return

def add_image(path, letter, arr):
	'''
	The process of adding a new image to the file.
	The path direct to the fill.
	The letter is given as well as the 16x16 8 bits image of it
	'''
	line = "{},{}\n".format(letter, ','.join(str(n) for n in arr.reshape(-1)))
	f = open(path, "a")
	f.write(line)
	f.close()
	return

def shuffle_textfile(path):
	'''
	Shuffle the lines of a text file
	'''
	lines = open(path).readlines()
	random.shuffle(lines)
	open(path, 'w').writelines(lines)
	return

def update_tally(path):
	tally = {
		"A":0,"B":0,"C":0,"D":0,"E":0,
		"F":0,"G":0,"H":0,"I":0,"J":0,
		"K":0,"L":0,"M":0,"L":0,"N":0,
		"O":0,"P":0,"Q":0,"R":0,"S":0,
		"T":0,"U":0,"V":0,"W":0,"X":0,
		"Y":0,"Z":0,}

	# complete the count
	reader = open(path, 'r')
	line = reader.readline()
	while line:
		letter = line.split(",")[0] # the letter of the line
		tally[letter] += 1          # add to the count
		line = reader.readline()    # next line
	reader.close()

	# update the js
	for letter in tally:
		display(Javascript('document.getElementById("letter-' + letter + '-count").innerHTML=' + str(tally[letter]) ))
	return


#######################################################################################################################################
### METHODS TO DISPLAY THE TEXTFILE ###
#######################################

def translate_line(line):
  arr = np.array([int(i) for i in line.split(',')[1:]]).reshape(16,16).astype(np.uint8)
  b64 = ndarray_to_base64(arr)
  b64 = "data:image/jpg;base64," + str(b64)[2:-1]
  return line[0], b64
  
def photo_html(num, b64, letter):
  return '''
<div class="photo">
  Line n°{}
  <div class="background">
    <img class="stretch" src="{}">
  </div>
  Letter : {}
</div>
'''.format(num,b64,letter)

def print_textfile(path):
	display(HTML('''
<html>
	<head>
		<style>
.photo {
	position: relative;
	float: left;
	width: 100px;
}
.background {
	width: 100px;
	height: 100px; 
	left: 0px; 
	top: 0px; 
	z-index: -1;
}
.stretch {
	width:100%;
	height:100%;
}
#gallery {
	height: 150px;
	width: 1000px;
}
		</style>
	</head>
	<body>
		<div id="gallery"></div>
	</body>
</html>
	'''))
	f = open(path,'r')
	lines = f.readlines()
	for n,line in enumerate(lines):
		letter, b64 = translate_line(line)
		display(HTML(photo_html(n+1, b64, line[0])))
		pass
	f.close()
	return

#######################################################################################################################################
### THE APPS ###
################


#######################################################################################################################################
### THE PHOTO BOOTH APP ###
###########################

class App_photobooth:
	'''
	The functions of this class need to be called in a certain context, 
	were the appropriate html and js code has been loaded.
	'''
	# HTML location on github
	HTML_URL = "https://raw.githubusercontent.com/Erffa/Automatic-Signal-Detector/master/photobooth.html"
	# FACEDETECT the path to the classifier end the classifier
	URL_FACE_CC = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
	FACE_CC = cv2.CascadeClassifier(cv2.samples.findFile(URL_FACE_CC))
	# CAMSHIFT
	TERM_CRIT = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	COLOR1 = np.array((0., 60., 32.)) # np.array((25., 76., 102.))
	COLOR2 = np.array((180., 255., 255.)) # np.array((125., 130., 204.))
	# SAVE IMAGES
	LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
  
	js = Javascript('''
// save / reset last pressed key
var code = false;
document.addEventListener('keypress', logKey);  
function logKey(e) { code = e.key.toUpperCase(); }
function resetCode(e) {
	let memo = code;
	code = false;
	return memo;
}
// functions and vaiables to know when strat/stop camshift
var camshift_is_on = false;
document.getElementById('camshift-but-start').onclick = async function () {
	if (camshift_is_on) { console.log("Camshift already running"); }
	else {
		camshift_is_on = true;
		invoke = google.colab.kernel.invokeFunction;
		result = await invoke("camshift", [], {});
	}
}
document.getElementById('camshift-but-stop').onclick = function () { camshift_is_on = false; }
// functions to start stop the HSV routine
var hsv_is_on = false;
document.getElementById("HSV-but-start").onclick = async function () {
	if (hsv_is_on) { console.log("HSV is already running"); }
	else {
		hsv_is_on = true;
		invoke = google.colab.kernel.invokeFunction;
		result = await invoke("hsv", [], {});
	}
}
document.getElementById("HSV-but-stop").onclick = function () { camshift_is_on = false; }
// Return pic dimensions

''')
	def __init__(self, savedir, textfile):
		#
		self.savedir = savedir
		self.textfile = textfile
		#
		self.full_box = np.array([0,0,640,480]) #self.w,self.h])
		# dimensions of the image
		self.w = 640
		self.h = 480
		# 
		self.frame = None # the RGB webcam image
		# camshift variables
		self.facearea = None
		self.prob_box = None
		self.prob = None # probability field ; 
		self.ret = None
		return
		#

	def launch(self):
		# ignore the warnings to do not change the output
		warnings.filterwarnings("ignore")
		# set the height of the cell
		display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 680});'''))
		# register the function
		output.register_callback('camshift', self.camshift)
		# get the html ; a request to get the content of the url
		request_html = requests.get(App_photobooth.HTML_URL, allow_redirects=True)
		# decode and exec the code
		html = HTML(request_html.content.decode("utf-8"))
		# upload the html
		display(html)
		display(App_photobooth.js)
		# launch the webcam
		eval_js('start()')
		# update the tally
		update_tally(savedir+textfile)
		return
		
	def getimg(self):
		capt = eval_js('capture()')
		return  base64_to_ndarray( capt )

	def setimg(self, img):
		imgb64 = ndarray_to_base64(img)
		imgb64 = "data:image/jpg;base64," + str(imgb64)[2:-1]
		eval_js('showimg("{}")'.format(imgb64))
		return

	def facedetect(self, img, box):
		# selection
		img_ = cutter(img, box)
		# filter the input
		gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
		gray = cv2.equalizeHist(gray)
		# do the recognition with ada boost
		rects = App_photobooth.FACE_CC.detectMultiScale(
			gray, scaleFactor=1.3, minNeighbors=4,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
		# check if anything was found
		founded = len(rects)!=0
		# find best rectangle and box
		rect = self.choose_rect(rects)
		return founded, rect

	def choose_rect(self, rects):
		l=len(rects)
		if rects==():
			return None
		elif l==1:
			return rects[0,:]
		else:
			best_rect=rects[0,:]
			best_area=best_rect[2]*best_rect[3]
			for i in range(1,l):
				rect=rects[i,:]
				area=rect[2]*rect[3]
				if area>best_area:
					best_rect=rect
					best_area=area
					pass
				pass
			pass
			return best_rect

	def compute_hist(self, img, box):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv_ = cutter(hsv, box)
		mask_ = cv2.inRange(hsv_, self.color1, self.color2)
		hist = cv2.calcHist([hsv_],[0],mask_,[180],[0,180])
		cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
		return hist

	def camshift(self):
		###################
		# DETECT THE FACE #
		##############################################
		founded = False
		rect = None
		while not founded and eval_js('webcam_is_on()'):
			self.frame = self.getimg()
			self.vis = self.frame.copy()
			founded, rect = self.facedetect(self.frame, self.full_box)
			# show the webcam on the bigscreen
			self.setimg(self.vis)
			pass

		#########################
		# COMPUTE THE HISTOGRAM #
		#############################################
		# remember the face location
		self.rect_box = rect_to_box(rect)
		self.facearea = expend(self.rect_box, (-30,-30,30,50), self.w,self.h)
		# compute the histogram
		self.hist_box = expend(self.rect_box, (-30,-30), self.w,self.h) # take a smaller area inside the facedetection area (no hair nor background)
		self.hist = self.compute_hist(self.frame, self.hist_box)
		# show boxes
		self.vis = draw_box(self.vis, self.hist_box, (255,0,0))
		self.vis = draw_box(self.vis, self.rect_box, (0,255,0))
		self.vis = draw_box(self.vis, self.facearea, (0,0,255))
		self.setimg(self.vis)

		#################
		# CAMSHIFT LOOP #
		#############################################
		self.search_area = self.full_box
		ison = eval_js('webcam_is_on()') and eval_js('camshift_is_on')
		while (ison):
			# get the image
			self.frame = self.getimg()
			self.vis = np.copy(self.frame)
			# transform the image
			self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
			self.mask = cv2.inRange(self.hsv, App.COLOR1, App.COLOR2)
			#
			self.prob = cv2.calcBackProject([self.hsv], [0], self.hist, [0, 180], 1)
			self.prob &= self.mask
			#
			x0,y0, x1,y1 = self.facearea
			self.prob[y0:y1,x0:x1] = 0

			self.ret, self.track_window = cv2.CamShift(self.prob, self.search_area, App.TERM_CRIT)
			self.track_box = rect_to_box(self.track_window)
			self.search_area = expend(self.track_box, (60,60), self.w,self.h)

			# if not goo enough, search everywhere
			quality = np.mean(cutter(self.prob, self.track_box))
			#print("{}".format(quality))
			if quality<20:
				self.search_area = self.full_box
				pass

			# affichage
			'''
			draw_box(self.vis, self.facearea, (6,0,0))
			draw_box(self.vis, self.track_window, (221,45,67))
			cv2.ellipse(self.vis, self.track_box, (0, 0, 255), 2)
			'''
			self.vis = self.prob.copy()
			self.vis = draw_box(self.vis, self.search_area, (200))
			self.setimg(self.vis)

			# test if a key has been press
			code = eval_js('resetCode()')
			if code != False:
				self.savepic(code)
				pass

			# Test if we go on
			ison = eval_js('webcam_is_on()') and eval_js('camshift_is_on')
			pass
		pass
		return
	

	def savepic(self, letter):
		
		# only letters
		if not letter in App_photobooth.LETTERS:
			return

		# cut the image, make it square
		self.square_box = centered_square_box(self.search_area)
		img = cutter(self.prob, self.square_box)

		# get the number
		l = os.listdir(self.savedir)
		idx = 1
		while "{}_{}_{}.jpg".format(letter, idx, 16) in l:
			idx += 1
			pass

		# save the images
		name16 = "{}_{}_16.jpg".format(letter, idx)
		name224 = "{}_{}_224.jpg".format(letter, idx)

		img16 = cv2.resize(img,(16,16))
		img224 = cv2.resize(img,(224,224))

		cv2.imwrite(self.savedir + name16, img16)
		cv2.imwrite(self.savedir + name224, img224)

		# add to file, update tally
		add_image(self.savedir+self.textfile, letter, img16)
		update_tally(self.savedir+self.textfile)

		return

#######################################################################################################################################
### THE ANALYST APP ###
#######################
	


def load_dataset(dataset_file_path):
    a = np.loadtxt(dataset_file_path, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    return a[:,0], a[:,1:]

class App_analyst:

	DEFAULT_INPUT_PATH = "/content/gdrive/My Drive/Colab Notebooks/storage/dataset2.txt"

	js = Javascript('''
// usefull constant
var alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split('');
invoke = google.colab.kernel.invokeFunction;
var gallery = document.getElementById("gallery");
// SET THE PATH!
var path_btn = document.getElementById("path-btn");
var path_input = document.getElementById("path-input");
path_input.value = "/content/gdrive/My Drive/Colab Notebooks/storage/dataset2.txt"
path_btn.onclick = async function () {
	await invoke("setter", [], {"path":path_input.value} );
}
// DELETE IMAGES
document.getElementById("btn-yes").onclick = async function () {
  await invoke("delete_line", [], {"line":this.line});
  modal.style.display = "none";
}

var gallery = document.getElementById("gallery");
''')
  
	def __init__(self, path=None):
		self.path = path
		return

	def launch(self):
		'''
		Launch the App object. 
		Call to build to create the html and add some the
		function registration 
		'''
		# create the interface in the cell
		self.build()
		# set maximal height of the cell
		display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 680});'''))
		# register the functions needed
		output.register_callback('setter', self.setter)
		output.register_callback('delete_line', self.delete_line)
		return

	@staticmethod
	def run(path=None):
		'''
		Create and launch an App object. Meant to be use as a solo statement in a cell
		'''
		app = App_analyst(path)
		app.launch()
		return

	def build(self):
		'''
		Create all the html and js needed
		'''
		display(html)
		display(App_analyst.js)
		self.refresh_gallery()
		return

	def refresh_gallery(self, **kwars):
		'''
		Recreate the gallery element with all the images from the dataset
		'''
		# if path is None, nothing to do
		if self.path == None:
			return

		# empty the current image displayer
		display(Javascript('''gallery.innerHTML = ""; '''))

		# extract the data from the textfile
		letters, arr = load_dataset(self.path)
		all_letters = np.unique(letters).astype(np.uint8)

		# create a row for each letter represented in the textfile 
		for letter in all_letters:
			App_analyst.HTML_add_row(letter)
			pass

		# add every image to the corresponding row
		for i in range(len(letters)):
			letter = int(letters[i])
			url = "data:image/jpg;base64," + str( ndarray_to_base64( arr[i].reshape((16,16)).astype(np.uint8) ) )[2:-1]
			App_analyst.HTML_add_image(i, letter, url)
			pass

		return

	@staticmethod
	def HTML_add_row(letter):
		display(Javascript('''
			let row = document.createElement("div");
			let title = document.createElement("div");
			let handler = document.createElement("div");

			row.id = "row-{l}";
			row.className = "row";
			title.id = "row-title-{l}";
			title.className = "title";
			title.innerHTML = alphabets[{l}];
			handler.id = "letter-handler-{l}";
			handler.className = "letter-handler";

			row.appendChild(title);
			row.appendChild(handler);
			gallery.appendChild(row);
			'''.format(l=letter)))
		return
  
	@staticmethod
	def HTML_add_image(index, letter, url):
		display(Javascript('''
			let handler = document.getElementById("letter-handler-{l}");
			let panel = document.createElement("div");
			let image = document.createElement("img");

			panel.id = "panel-{l}";
			panel.className = "background";
			image.id = "image-{i}";
			image.className = "stretch";
			image.src = "{u}";
			image.onclick = function () {bl}
			num = parseInt(this.id.split("-")[1]);
			modal.style.display = "block";
			document.getElementById("confirm-modal-image").src = this.src;
			document.getElementById("btn-yes").line = num;
			{br}

			panel.appendChild(image);
			handler.appendChild(panel);
			'''.format(i=index, l=letter, u=url, bl="{", br="}")))
		return

	def setter(self, **kwargs):
		if "path" in kwargs.keys():
			self.path = kwargs["path"]
			self.refresh_gallery()
			pass
		return

	def delete_line(self, **kwargs):
	if "line" in kwargs.keys():
		# get the index of the line to delete
		line = kwargs["line"]
		# get all the lines
		f = open(self.path, 'r')
		lines = f.readlines()
		f.close()
		# rewrite the file
		f = open(self.path, 'w')
		for i in range(len(lines)):
			if i != line:
				f.write(lines[i])
				pass
			pass
		f.close()
		# refresh the display
		self.refresh_gallery()
		pass
	return
