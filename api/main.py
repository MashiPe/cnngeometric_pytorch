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

sys.path.append(str(Path.cwd()))

from stitching.stitch import stitch
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


    return jsonify({'message':'zip received','path':savePath})

@app.route('/api/stitch', methods=['POST'])
def stitchImgs():
    
    print('procecssing images')
    
    iduser = request.args['idusuario']
    scanId = request.args['scanid']

    zip_folder = path.join(home,'zips',iduser,scanId)
    imgs_folder = path.join(home,'uncompressed',scanId)

    if (not path.exists(imgs_folder)):
        makedirs(imgs_folder)

    unzipFiles(zip_folder,imgs_folder)

    
    file_paths = [f for f in listdir(imgs_folder) if path.isfile(path.join(imgs_folder, f))]

    file_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    file_paths = [ path.join(imgs_folder,f) for f in file_paths]

    dest_folder = path.join(home,'stitchedimgs',iduser,scanId)

    filePath = stitch(file_paths,dest_folder)

    url = '{}:5000/api/getimg?idusuario={}&scanid={}'.format(HOSTNAME,iduser,scanId)

    print("stitch finish")

    return jsonify( { 'url':url } )

@app.route('/api/getimg')
def getimg():

    iduser = request.args['idusuario']
    scanid = request.args['scanid']

    imgPath = path.join(home,'stitchedimgs',iduser,scanid,'result.png')

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