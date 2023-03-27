from crypt import methods
from pathlib import Path
from flask import Flask, request, Response,jsonify,send_file
from flask_cors import CORS
# import numpy as np
# import cv2 as cv
from os import listdir, path, makedirs
import sys
import numpy as np
import uuid
import cv2 as cv
import pymongo

sys.path.append(str(Path.cwd()))

from stitching.stitch import stitch, ImageStitcher
from api.utils.zip_utils import unzipFiles
from werkzeug.serving import WSGIRequestHandler


home = str(Path.cwd())

home = path.join(home,'apifiles')
print(path.isdir(home))
print(home)


if (not path.isdir(home)):
    makedirs(home)



# HOSTNAME = 'http://redpanda.sytes.net'
HOSTNAME = 'http://173.255.114.112'
MONGODATABASE = 'tours'
MONGOUSERNAME = 'toursuser'
MONGOPASSWD = '3dspace@tours#cuenca#1417'
SCANSCOLLECTION = 'scans'

# MONGOHOST = 'mongodb://{}:{}@localhost:27017/'.format(MONGOUSERNAME,MONGOPASSWD)
MONGOHOST = 'mongodb://127.0.0.1:27017/'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = home

CORS(app)
@app.route('/api/test/stitch', methods=['POST'])
def test():
    
    print('procecssing images')
    files = request.files.getlist('files')
    print(files)
    file_paths = []

    for file in files:
        filename = file.filename
        fullPath = path.join(home,filename)
        file_paths.append(fullPath)
        file.save(fullPath)
        print(fullPath)


    stitch(file_paths)
    

    # build a response dict to send back to client
    response = {'message': 'images received. num={}'.format(len(file_paths))}

    return Response(response=response, status=200, mimetype="application/json")



@app.route('/api/scans',methods=['GET'])
def getScans():

    idUser = request.args['idusuario']
    # scanId = request.args['scanid']

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]

    res = {
        "scanList":[]
    }
    for scan in mycol.find({"userid":idUser}):
        print(scan['scanid'])
        res['scanList'].append(scan['scanid'])
        # pass
    
    return res

@app.route('/api/scans/remove',methods=['POST'])
def getScans():

    idUser = request.args['idusuario']
    scanId = request.args['scanid']

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]

    
    mycol.find({"userif":idUser,"scanid":scanId})
    
    return jsonify({'message':'Scand deleted'})

@app.route('/api/variants',methods=['GET'])
def getScanVariants():

    idUser = request.args['idusuario']
    scanId = request.args['scanid']

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]

    variants = {}
    for scanRegistry in mycol.find({"userid":idUser,"scanid":scanId}):
        variants = scanRegistry['stitchvariants']
        # pass
    
    return variants


@app.route('/api/save',methods=['POST'])
def saveImages():
    zipFile = request.files['file']
    
    idUser = request.args['idusuario']
    scanId = request.args['scanid']
    name = request.args['nombre']

    savePath = path.join(home,'zips',idUser,scanId)

    if (not path.exists(savePath)):
        makedirs(savePath)

    savePath = path.join(savePath,name)

    # savePath = path.join(home,'zips',idUser,idTour,)
    
    zipFile.save(savePath)

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]

    if len(list(mycol.find({"userid":idUser,"scanid":scanId})))==0:
        mycol.insert_one({'userid':idUser,"scanid":scanId,"stitchvariants":{}})
        

    return jsonify({'message':'zip received','path':savePath})

@app.route('/api/stitch/ensemble', methods=['POST'])
def stitchImgsEnsemble():

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]
    
    print('procecssing images')

    request.args.to_dict(flat=False)
    
    iduser = request.args['idusuario']
    scanId = request.args['scanid']
    # model = request.args['model']
    # model = model.split(',')
    prep_method = request.args['prep_method']
    x_axis_only = request.args['x_axis_only']

    zip_folder = path.join(home,'zips',iduser,scanId)
    imgs_folder = path.join(home,'uncompressed',scanId,'ensemble')

    data = request.get_json()

    ensemble = data['ensemble']

    if (not path.exists(imgs_folder)):
        makedirs(imgs_folder)

    unzipFiles(zip_folder,imgs_folder)


    # Adding variant to database
    query = {"userid":iduser,"scanid":scanId}

    registry = mycol.find_one(query)

    variants = registry['stitchvariants']

    variants['ensemblevariant'] = { 'completed':False }

    mycol.update_one(query,{"$set": {"stitchvariants":variants} })

    
    file_paths = [f for f in listdir(imgs_folder) if path.isfile(path.join(imgs_folder, f))]

    file_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    file_paths = [ path.join(imgs_folder,f) for f in file_paths]

    dest_folder = path.join(home,'stitchedimgs',iduser,scanId)

    # filePath = stitch(file_paths,dest_folder)

    ensemble_img_list = []

    for ensemble_queue in ensemble:

        stitcher = ImageStitcher(ensemble_queue,prep_method)
        imgList = stitcher.stitchv2(file_paths,dest_folder,returnImgsList=True,x_axis_only=x_axis_only)

        ensemble_img_list.append(imgList)
    
    for i in range(1,len(ensemble_img_list)):

        for j,_ in enumerate(ensemble_img_list[i]):

            aux_transform_stack = []
            for k,_ in enumerate(ensemble_img_list[i][j].transform_stack):
                
                new_el = ensemble_img_list[0][j].transform_stack[k] +ensemble_img_list[i][j].transform_stack[k] 

                aux_transform_stack.append(new_el)


            ensemble_img_list[0][j].transform_stack = aux_transform_stack

    for j, _ in enumerate(ensemble_img_list[0]):

        aux_transform_stack = []
        for k,_ in enumerate(ensemble_img_list[0][j].transform_stack):
            
            new_el = (ensemble_img_list[0][j].transform_stack[k])/len(ensemble_img_list) 

            aux_transform_stack.append(new_el)


        ensemble_img_list[0][j].transform_stack = aux_transform_stack

    pano = stitcher.transformAndBlend(ensemble_img_list[0],x_axis_only=x_axis_only)

    y_dim = pano.shape[0]
    x_dim = pano.shape[0]*2

    resized = cv.resize(pano,(x_dim,y_dim))

    # filePath_1 = path.join(dest,'result'+self.model_name+'.jpg')
    dest_folder = path.join(home,'stitchedimgs',iduser,scanId)
    filePath = path.join(dest_folder,'ensemblevariant.png')
    
    cv.imwrite(filePath,resized)    


    url = '{}:5000/api/getimg?idusuario={}&scanid={}'.format(HOSTNAME,iduser,scanId)

    print("stitch finish")
    
    #Updating variant status
    
    registry = mycol.find_one(query)

    variants = registry['stitchvariants']

    variants['ensemblevariant'] = { 'completed':True }

    mycol.update_one(query,{"$set": {"stitchvariants":variants} })

    return jsonify( { 'url':url } )

@app.route('/api/stitch', methods=['POST'])
def stitchImgs():

    myclient = pymongo.MongoClient(MONGOHOST)
    mydb = myclient[MONGODATABASE]
    mycol = mydb[SCANSCOLLECTION]
    
    print('procecssing images')

    request.args.to_dict(flat=False)
    
    iduser = request.args['idusuario']
    scanId = request.args['scanid']
    model = request.args['model']
    model = model.split(',')
    prep_method = request.args['prep_method']
    x_axis_only = request.args['x_axis_only']

    zip_folder = path.join(home,'zips',iduser,scanId)
    imgs_folder = path.join(home,'uncompressed',scanId)

    if (not path.exists(imgs_folder)):
        makedirs(imgs_folder)

    unzipFiles(zip_folder,imgs_folder)

    # Adding variant to database
    query = {"userid":iduser,"scanid":scanId}

    registry = mycol.find_one(query)

    variants = registry['stitchvariants']

    variants['{}variant'.format(model[0])] = { 'completed':False }

    mycol.update_one(query,{"$set": {"stitchvariants":variants} })

    # Starting stitching process    
    file_paths = [f for f in listdir(imgs_folder) if path.isfile(path.join(imgs_folder, f))]

    file_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    file_paths = [ path.join(imgs_folder,f) for f in file_paths]

    dest_folder = path.join(home,'stitchedimgs',iduser,scanId)

    if (not path.exists(dest_folder)):
        makedirs(dest_folder)
    # filePath = stitch(file_paths,dest_folder)

    stitcher = ImageStitcher(model,prep_method)
    pano = stitcher.stitchv2(file_paths,dest_folder,x_axis_only=x_axis_only)

    # filePath_1 = path.join(dest,'result'+self.model_name+'.jpg')
    filePath = path.join(dest_folder,'{}variant.png'.format(model[0]))
    
    cv.imwrite(filePath,pano)    
    # cv.imwrite(filePath_1,resized)    

    url = '{}:5000/api/getimg?idusuario={}&scanid={}'.format(HOSTNAME,iduser,scanId)

    print("stitch finish")
    
    #Updating variant status
    
    registry = mycol.find_one(query)

    variants = registry['stitchvariants']

    variants['{}variant'.format(model[0])] = { 'completed':True }

    mycol.update_one(query,{"$set": {"stitchvariants":variants} })

    return jsonify( { 'url':url } )


@app.route('/api/scanvariant')
def getimg():

    iduser = request.args['idusuario']
    scanid = request.args['scanid']
    variant = request.args['variant']

    imgPath = path.join(home,'stitchedimgs',iduser,scanid,'{}.png'.format(variant))

    return send_file(imgPath, mimetype='image/png')

@app.route('/api/tours/getresultado')
def getresultado():

    iduser = request.args['idusuario']
    idtour = request.args['idtour']
    scene = request.args['escena']
    img = request.args['imagen']

    imgPath = path.join('/images',iduser,idtour,scene,img,'result.jpg')

    return send_file(imgPath, mimetype='image/png')


WSGIRequestHandler.protocol_version = "HTTP/1.1"

# start flask app
app.run(host="0.0.0.0", port=5000)