import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
# Data for prediction
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier

# sidebar title
st.sidebar.write('Enter dataset to analyse')

# image source selection
option = st.sidebar.selectbox('Select upload method', ['Use a sample dataset', 'Use your own dataset'])
valid_datasets = glob.glob('data/*.csv')

if option == 'Use a sample dataset':
    st.sidebar.write('Select a sample dataset')
    fname = st.sidebar.selectbox('Select from existing list',
                                 valid_datasets)

else:
    st.sidebar.write('Select an dataset to upload')
    fname = st.sidebar.file_uploader('Choose a CSV file to upload',
                                     type=['csv'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_datasets[0]

    # Can be used wherever a "file-like" object is accepted:
df = pd.read_csv(fname)
columns = df.columns

def page0():
    st.write(f'Show dataset of {fname}')
    st.write(df)
def page1():
    optChart = ["Line chart","Histogram","Box plot","Scatter plot","Contour plot","Heat map"]
    ch = st.selectbox("Select chart type",optChart)
    if ch == "Line chart":
        fea = st.multiselect("Select feature",columns)
        for f in fea:
            fig, axes = plt.subplots(figsize=(15, 5))
            sns.lineplot(df[f], ax=axes)
            st.pyplot(fig)
    if ch == "Scatter plot":
        fea = st.multiselect("Select feature",columns)
        target = st.selectbox("Select target",columns)
        fig = plt.figure(figsize=(12,9))
        sns.scatterplot(x=fea[0], y=fea[1], data=df, size=df[target],hue=df[target])
        st.pyplot(fig)

    if ch == "Histogram":
        fea = st.multiselect("Select feature",columns)
        for f in fea:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.histplot(df[f], kde=True, ax=axes)
            st.pyplot(fig)
    if ch == "Box plot":
        fea = st.multiselect("Select feature",columns)
        for f in fea:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df,x = df[f], ax=axes)
            st.pyplot(fig)
    if ch == "Heat map":
        fea = st.multiselect("Select feature",columns)
        corr_val = df[fea].corr().round(2)
        fig, axes = plt.subplots(figsize=(8, 8))
        sns.heatmap(corr_val, annot=True, cmap='rainbow') #YlGnBu
        st.pyplot(fig)
    if ch == "Contour plot":
        from scipy.interpolate import griddata
        fea = st.multiselect("Select feature",columns)
        target = st.selectbox("Select target",columns)

        def plot_contour(df,featx,featy,featz,npts=100):
            fig = plt.figure(figsize=(9,6))
            x = df[featx]
            y = df[featy]
            z = df[featz]
            r_x = x.max()-x.min()
            r_y = y.max()-y.min()
            x_max ,x_min= x.max()+0.01*r_x,x.min()-0.01*r_x
            y_max ,y_min= y.max()+0.01*r_y,y.min()-0.01*r_y
            xi = np.linspace(x_min,x_max,npts)
            yi = np.linspace(y_min,y_max,npts)
            # grid the data.
            # contour the gridded data, plotting dots at the randomly spaced data points.
            zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear') # nearest ,linear
            tile = "Between {} and {}".format(featx,featy)
            plt.contour(xi,yi,zi,linewidths=1.5,cmap="jet")
            plt.colorbar() # draw colorbar
            plt.xlim(x_min,x_max)
            plt.ylim(y_min,y_max)
            plt.xlabel(featx)
            plt.ylabel(featy)
            plt.title(tile)
            return fig

        fig = plot_contour(df,fea[0],fea[1],target,npts=100)
        st.pyplot(fig)

def page2():
    from sklearn.preprocessing import StandardScaler,minmax_scale
    from sklearn.model_selection import train_test_split
    fea = st.multiselect("Select feature",columns)
    target = st.selectbox("Select target",columns)
    sc = StandardScaler()
    X = df[fea]
    y = df[target]
    X = sc.fit_transform(X)

    (X, Xt, y, yt) = train_test_split(X, y,
        test_size=0.25, random_state=101)

    options = ["Linear(Poly)Regression","KNeighborsRegression","DecisionTreeRegression","RandomForestRegression","Keras ANN Regression"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(options)
    with tab1:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_squared_error, r2_score
        degree = st.slider("Degree",1,9)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        Xt_poly = poly.transform(Xt)
        lm = LinearRegression()
        lm.fit(X_poly,y)
        y_pred = lm.predict(X_poly)
        yt_pred = lm.predict(Xt_poly)
        r2_train = lm.score(X_poly,y)
        r2_test = lm.score(Xt_poly,yt)
        rmse = np.sqrt(mean_squared_error(yt,yt_pred))
        coef = lm.coef_
        intc = lm.intercept_

        fig3 = plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(y, y_pred, alpha=0.5, color='blue')
        m, b = np.polyfit(y, y_pred, 1)
        plt.plot(y, m*y + b, color='pink')
        plt.xlabel('Actual Target Train', fontsize=14)
        plt.ylabel('Predicted Target Train', fontsize=14)
        plt.title('Linear regression Predicted VS. Actual Target Train', fontsize=16)
        plt.grid(linewidth=0.5)
        st.pyplot(fig3)
        st.write(f'r2_train ={r2_train}')
        st.write(f'r2_test ={r2_test}')
        st.write(f'rmse ={rmse}')
        st.write(f'intercept ={intc}')
        st.write(f'coef ={coef}')

    with tab2:
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.neighbors import KNeighborsRegressor
        k = st.number_input("n_neighbors",3,100,5,1)
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X,y)
        y_pred = knn.predict(X)
        yt_pred = knn.predict(Xt)
        r2_train = knn.score(X,y)
        r2_test = knn.score(Xt,yt)
        rmse = np.sqrt(mean_squared_error(yt,yt_pred))

        fig3 = plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(y, y_pred, alpha=0.5, color='blue')
        m, b = np.polyfit(y, y_pred, 1)
        plt.plot(y, m*y + b, color='pink')
        plt.xlabel('Actual Target Train', fontsize=14)
        plt.ylabel('Predicted Target Train', fontsize=14)
        plt.title('Knn regression Predicted VS. Actual Target Train', fontsize=16)
        plt.grid(linewidth=0.5)
        st.pyplot(fig3)
        st.write(f'r2_train ={r2_train}')
        st.write(f'r2_test ={r2_test}')
        st.write(f'rmse ={rmse}')

    with tab3:
        # Decision Tree
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        dt_reg = DecisionTreeRegressor(max_depth=5)
        model = dt_reg.fit(X,y)
        y_pred_rf = dt_reg.predict(X)
        ypt_rf = dt_reg.predict(Xt)
        r2_train_rf = dt_reg.score(X,y)
        r2_test_rf = dt_reg.score(Xt,yt)
        rmse = np.sqrt(mean_squared_error(yt,ypt_rf))

        fig5 = plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(y, y_pred_rf, alpha=0.5, color='blue')
        m, b = np.polyfit(y, y_pred_rf, 1)
        plt.plot(y, m*y + b, color='pink')
        plt.xlabel('Actual Target Train', fontsize=14)
        plt.ylabel('Predicted Target Train', fontsize=14)
        plt.title('Dicision tree Predicted VS. Actual Target Train', fontsize=16)
        plt.grid(linewidth=0.5)
        st.pyplot(fig5)
        st.write(f'r2_train ={r2_train_rf}')
        st.write(f'r2_test ={r2_test_rf}')
        st.write(f'rmse ={rmse}')

    with tab4:
        # Random forest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        rf_reg = RandomForestRegressor(max_depth=5)
        model = rf_reg.fit(X,y)
        y_pred_rf = rf_reg.predict(X)
        ypt_rf = rf_reg.predict(Xt)
        r2_train_rf = rf_reg.score(X,y)
        r2_test_rf = rf_reg.score(Xt,yt)
        rmse = np.sqrt(mean_squared_error(yt,ypt_rf))
        fig4 = plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(y, y_pred_rf, alpha=0.5, color='blue')
        m, b = np.polyfit(y, y_pred_rf, 1)
        plt.plot(y, m*y + b, color='pink')
        plt.xlabel('Actual Target Train', fontsize=14)
        plt.ylabel('Predicted Target Train', fontsize=14)
        plt.title('Random forest Predicted VS. Actual Target Train', fontsize=16)
        plt.grid(linewidth=0.5)
        st.pyplot(fig4)
        st.write(f'r2_train ={r2_train_rf}')
        st.write(f'r2_test ={r2_test_rf}')
        st.write(f'rmse ={rmse}')

    with tab5:
        from keras.models import Sequential
        from keras.layers import Dense
        epochs = st.number_input("Enter epoch no",5,200,30,5)
        layer = st.text_input("Enter layer","128,64,32")
        l = list(map(int,layer.split(",")))
        # print(l)

        def create_model(input_dim,layer=[64,32]):
            model = Sequential()
            model.add(Dense(layer[0], input_dim=input_dim, activation='relu'))
            for l in layer[1:]:
                model.add(Dense(l, activation='relu'))
            model.add(Dense(1, activation='linear'))
            return model
        
        train = st.button("Train Model")
        label_res = st.empty()
        label_res.text("คุณยังไม่ได้กดปุ่ม Train")
        if train:
            model1 = create_model(X.shape[1], l)
            model1.compile(loss='mean_squared_error',
                            optimizer='adam',
                            metrics=['mae'])
            history_callback = model1.fit(X, y,
                                        batch_size=8,
                                        epochs=epochs,
                                        verbose='auto',
                                        validation_data=(Xt, yt),
                                        )
            yt_pred = model1.predict(Xt)
            y_pred = model1.predict(X)
            score = model1.evaluate(Xt, yt, verbose=0)
            r2_train = r2_score(y,y_pred)
            r2_val = r2_score(yt,yt_pred)

            fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12, 9))

            val_accuracy = history_callback.history['val_mae']
            val_loss = history_callback.history['val_loss']
            train_accuracy = history_callback.history['mae']
            train_loss = history_callback.history['loss']

            ax1.plot(train_accuracy, label='mae')
            ax1.plot(val_accuracy, label='val_mae')
            ax2.plot(train_loss, label='loss')
            ax2.plot(val_loss, label='val_loss')
            ax1.set_ylabel('accuracy(mae)')
            ax2.set_ylabel('loss')
            ax2.set_xlabel('epochs')
            ax1.legend()
            ax2.legend()

            st.pyplot(fig)
            st.write(f'r2_train ={r2_train}')
            st.write(f'r2_test ={r2_val}')
            st.write(f'Test mae ={score[1]}')
            st.write(f'Test loss ={score[0]}')

def page3():
    import cv2
    import numpy as np
    st.markdown(f" {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo illustrates the use of the st.camera_input widget 
        — which lets the user take an image through their webcam and upload it to the app 
        — to apply a filter to the uploaded image. Enjoy!
        """
    )
    def preprocess(img):
        bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        return img

    def invert(img):
        img = preprocess(img)
        inv = cv2.bitwise_not(img)
        return inv

    def sketch(img):
        img = preprocess(img)
        _, sketch_img = cv2.pencilSketch(
            img, sigma_s=60, sigma_r=0.07, shade_factor=0.1
        )
        return sketch_img
    
    def edge(img):
        img = preprocess(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # burr = cv2.GaussianBlur(gray_img,(7,7),0)
        edged = cv2.Canny(gray_img,80,180)
        gray_img = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
        return gray_img
    
    def gray(img):
        img = preprocess(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        return gray_img

    def none(img):
        img = preprocess(img)
        return img

    picture = st.camera_input("First, take a picture...")

    filters_to_funcs = {
        "No filter": none,
        "Grayscale": gray,
        "Invert": invert,
        "Sketch": sketch,
        "Edge detect": edge,
    }
    filters = st.selectbox("...and now, apply a filter!", filters_to_funcs.keys())

    if picture:
        st.image(filters_to_funcs[filters](picture), channels="BGR")

def page4():
    import cv2
    import numpy as np
    from yolov8_detect import yolov8_detect,draw_detections

    st.markdown(f" {list(page_names_to_funcs.keys())[4]}")
    st.write(
        """
        This is casting defect detection beta version to the uploaded image 
        and predected casting defect. Enjoy it!
        """
    )
    # from yolo_predictions import YOLO_Pred
    # yolo = YOLO_Pred('best.onnx','defect1301.yaml')
    classes = ['Dent','PinHole','Gas','Slag','Shrinkage','SandDrop','SandBroken','Other']
    method = ["Demo image","Up load from file","Webcam image"]
    select = st.selectbox("Please select image input method",method)
    img = None
    if select=="Demo image":
        imglist = ['data/test.jpg','data/test1.jpg','data/test2.jpg',
                   'data/test3.jpg','data/test4.jpg','data/test5.jpg','data/test6.jpg','data/test7.jpg']
        ch = st.selectbox('Select image file from list',imglist)
        img = cv2.imread(ch)
    if select=="Up load from file":
        img_file = st.file_uploader("เปิดไฟล์ภาพ")
        if img_file is not None:    
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if select=="Webcam image":
        picture = st.camera_input("First, take a picture...")
        if picture is not None:
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        v8bbox, v8conf, v8class ,img = yolov8_detect(img)
        img = draw_detections(img,v8bbox,v8conf,v8class)
        # print(v8class)
        if len(v8class) > 0: #เจอวัตถุใน 20 อย่างนั้น
            obj_names = ''
            for c in v8class:
                obj_names += classes[c]+', '
            
            text_obj = 'ตรวจพบ ' + obj_names
        else:
            text_obj = 'ไม่พบ defect'
        #----------------------------------------------
        st.image(img, caption='ภาพ Output',channels="BGR")
        st.text(text_obj)

def page5():
    import cv2
    import numpy as np
    import glob

    st.markdown(f" {list(page_names_to_funcs.keys())[5]}")
    st.write(
        """
        This is image analyser for nodularity beta version to the uploaded image 
        and predected nodularity. Enjoy it!
        """
    )

    method = ["Demo image","Up load from file","Webcam image"]
    select = st.selectbox("Please select image input method",method)
    img = None
    img_files = ["data/n57.jpg","data/n90.jpg","data/n92.jpg"]
    if select=="Demo image":
        ch = st.selectbox("Select demo image",img_files)
        img = cv2.imread(ch)
    if select=="Up load from file":
        img_file = st.file_uploader("เปิดไฟล์ภาพ")
        if img_file is not None:    
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if select=="Webcam image":
        picture = st.camera_input("First, take a picture...")
        if picture is not None:
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blurr = cv2.GaussianBlur(gray,(9,9),0)
        (T,treaInv) = cv2.threshold(blurr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        (cnts, _) = cv2.findContours(treaInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        graphite = img.copy()
        gcnt = 0
        gptype = {"I" :0,"II" :0,"III" :0,"IV" :0,"V" :0,"VI" :0}
        colors = ()
        gpt = ""
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 20 :
                M = cv2.moments(c)
                perimeter = cv2.arcLength(c,True)
                Cx = int(M['m10']/M['m00'])
                Cy = int(M['m01']/M['m00'])
                circulality = 4*3.14159*area/(perimeter**2)
                if circulality <= 0.2 :
                    gpt = "I"
                    gptype[gpt] +=1
                    colors = (0,0,255)
                elif  circulality > 0.2 and circulality <= 0.3 :
                    gpt = "II"
                    gptype[gpt] +=1
                    colors = (255,0,255)
                elif  circulality > 0.3 and circulality <= 0.5 :
                    gpt = "III"
                    gptype[gpt] +=1
                    colors = (0,255,255)   
                elif  circulality > 0.5 and circulality <= 0.65 :
                    gpt = "IV"
                    gptype[gpt] +=1
                    colors = (255,255,0)   
                elif  circulality > 0.65 and circulality < 0.8 :
                    gpt = "V"
                    gptype[gpt] +=1
                    colors = (0,255,0)
                else :
                    gpt = "VI"
                    gptype[gpt] +=1
                    colors = (255,0,0)    
                gcnt +=1
                
                print('No. {:3} Type : {:5}, area = {:.2f} ,Perimeter = {:.2f},NodulaFactor = {:.2f}'.format(gcnt,gpt,area,perimeter,circulality) )
                cv2.drawContours(graphite, [c], -1, colors, -1)
        # nodul = (0.05*gptype["II"]+0.2*gptype["III"]+0.4*gptype["IV"]+0.9*gptype["V"]+gptype["VI"])/gcnt
        nodul = (gptype["V"]+gptype["VI"])/gcnt
        #----------------------------------------------
        col1,col2 = st.columns(2)
        col1.image(img, caption='ภาพ input',channels="BGR")
        col2.image(graphite, caption='ภาพ output',channels="BGR")
        # st.image(img, caption='ภาพ Output',channels="BGR")
        st.write(f'Type I : {gptype["I"]} ,Type II : {gptype["II"]},\nType III : {gptype["III"]} ,Type IV : {gptype["IV"]},\nType V : {gptype["V"]} ,Type VI : {gptype["VI"]}')
        st.text('Nodularity = '+str(nodul))
        st.text('Nodule count = '+str(gcnt))

page_names_to_funcs = {
    "Data frame": page0,
    "Data Visualized": page1,
    "Machine learning": page2,
    "Webcam picture": page3,
    "AI Casting defect": page4,
    "Nodularity analyse": page5,
}

list_name = st.sidebar.radio("## Choose a function", page_names_to_funcs.keys())
page_names_to_funcs[list_name]()