
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from io import BytesIO




def colorizer(img):
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    protxt=r"C:\Users\ullek\Downloads\colorization_deploy_v2.prototxt"
    model=r"C:\Users\ullek\Downloads\colorization_release_v2.caffemodel"
    kernel=r"C:\Users\ullek\Downloads\pts_in_hull.npy"
    net=cv2.dnn.readNetFromCaffe(protxt,model)
    points=np.load(kernel)


    points=points.transpose().reshape(2,313,1,1) 
    net.getLayer(net.getLayerId("class8_ab")).blobs=[points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs=[np.full([1, 313],2.606,dtype="float32")]
    normalized=img.astype("float32") / 255.0
    lab=cv2.cvtColor(normalized,cv2.COLOR_BGR2LAB)

    resized=cv2.resize(lab,(224,224))
    L=cv2.split(resized)[0]
    L-=50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab=net.forward()[0, :, :, :].transpose((1,2,0))
    ab=cv2.resize(ab,(img.shape[1],img.shape[0]))

    L=cv2.split(lab)[0]
    colorized=np.concatenate((L[:,:,np.newaxis],ab),axis=2)
    colorized=cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
    colorized=np.clip(colorized,0,1)

    colorized=(255.0 *colorized).astype("uint8")
    return colorized
    



file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
st.title("Colorize Black and White Image")
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    st.text("Your colorized image")
    color = colorizer(img)
    
    st.image(color, use_column_width=True)

    
    result = Image.fromarray(color)

    buf = BytesIO()
    result.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    
    btn = st.download_button(
      label="Download Image",
      data=byte_im,
      file_name="imagename.png",
      mime="image/jpeg",
      )



