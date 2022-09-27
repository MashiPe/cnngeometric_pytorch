from crypt import methods
from pathlib import Path
from flask import Flask, request, Response,jsonify,send_file
from flask_cors import CORS
# import numpy as np
# import cv2 as cv
from os import listdir, path, makedirs
import sys
import numpy as np

sys.path.append(str(Path.cwd()))

from stitching.stitch import stitch
from api.utils.zip_utils import unzipFiles

home = str(Path.cwd())

home = path.join(home,'apifiles')
print(path.isdir(home))
print(home)


if (not path.isdir(home)):
    makedirs(home)


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
    idTour = request.args['idtour']
    name = request.args['nombre']
    scene = request.args['escena']
    img = request.args['imagen']

    savePath = path.join(home,'zips',idUser,idTour,scene,img)

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
    idtour = request.args['idtour']
    scene = request.args['escena']
    img = request.args['imagen']

    zip_folder = path.join(home,'zips',iduser,idtour,scene,img)
    imgs_folder = path.join(home,'uncompressed',iduser,idtour,scene,img)

    if (not path.exists(imgs_folder)):
        makedirs(imgs_folder)

    unzipFiles(zip_folder,imgs_folder)

    
    file_paths = [f for f in listdir(imgs_folder) if path.isfile(path.join(imgs_folder, f))]

    file_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    file_paths = [ path.join(imgs_folder,f) for f in file_paths]

# files = request.files.getlist('files')
    # print(files)
    # file_paths = []

    # for file in files:
    #     filename = file.filename
    #     fullPath = path.join(home,filename)
    #     file_paths.append(fullPath)
    #     file.save(fullPath)
    #     print(fullPath)

    dest_folder = path.join('/images',iduser,idtour,scene,img)

    filePath = stitch(file_paths,dest_folder)

    url = 'http://localhost:5000/api/getimg?idusuario={}&idtour={}&escena={}&imagen={}'.format(iduser,idtour,scene,img)

    print("stitch finish")

    # build a response dict to send back to client
    # response = {'message': 'image stitched','url': url}

    # return jsonify(response)

    return send_file(filePath,mimetype='image/png')


@app.route('/api/tours/getresultado')
def myapp():

    iduser = request.args['idusuario']
    idtour = request.args['idtour']
    scene = request.args['escena']
    img = request.args['imagen']

    imgPath = path.join('/images',iduser,idtour,scene,img,'result.jpg')

    return send_file(imgPath, mimetype='image/png')

# start flask app
app.run(host="0.0.0.0", port=5000)