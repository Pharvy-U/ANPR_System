import time
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import socket
import pickle
import struct
from turbo_flask import Turbo
import threading

app = Flask(__name__)
turbo = Turbo(app)

blank = np.zeros((250, 500))
cam = blank
plate_cam = blank
plate_num = ''
oneitem = None

app.secret_key = " "

# SqlAlchemy Database Configuration With Mysql

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crud'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# Creating model table for our CRUD database
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    Model = db.Column(db.String(100))
    plate = db.Column(db.String(100))

    def __init__(self, name, Model, plate):
        self.name = name
        self.Model = Model
        self.plate = plate


# This is the index route where we are going to
# query on all our employee data

@app.route('/')
def home():
    try:
        back.start()
    except RuntimeError:
        pass
    return render_template("home.html")


@app.route('/tindex')
def Index():
    all_data = Data.query.all()

    return render_template("tindex.html", Lisence=all_data)


@app.route('/services')
def services():
    return render_template("services.html")


# this route is for inserting data to mysql database via html forms
@app.route('/insert', methods=['POST'])
def insert():
    if request.method == 'POST':
        name = request.form['name']
        Model = request.form['Model']
        plate = request.form['plate']

        my_data = Data(name, Model, plate)
        db.session.add(my_data)
        db.session.commit()

        flash("Vehicle Inserted Successfully")

        return redirect(url_for('Index'))


@app.context_processor
def reboot():
    global oneitem
    oneitem = Data.query.filter_by(plate=plate_num).first()
    return {'oneitem': oneitem, 'plating': plate_num}


@app.route('/gallery')
def gallery():
    return render_template("gallery.html")


def gen():
    global cam, plate_cam, plate_num
    IP = "192.168.43.78"    # IP address of server computer (Leave blank if hosted on a clcoud linux server)
    PORT = 9999
    address = (IP, PORT)

    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # create server socket
    ss.bind(address)
    ss.listen(5)
    print('listening...')
    data = b""
    payload_size = struct.calcsize("Q")
    connect = True
    while connect:
        print('Waiting for Connection...')
        cs, addr = ss.accept()
        print('Connected')
        while True:
            while len(data) < payload_size:
                try:
                    packet = cs.recv(4 * 1024)
                except ConnectionResetError:
                    connect = False
                    cs.close()
                    break
                if not packet:
                    break
                data += packet
            if not connect:
                connect = True
                break
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += cs.recv(4 * 1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)
            cam = frame['video']
            plate_cam = frame['plate roi']
            num = frame['plate number']
            if plate_num != num:
                plate_num = num
                print(plate_num)
            if oneitem:
                # print('Vehicle is registered')
                cs.send(bytes('Registered', 'utf-8'))
            if not oneitem:
                # print('Not Registered')
                cs.send(bytes('Not Registered', 'utf-8'))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cs.close()


back = threading.Thread(target=gen)


def live_vid():
    while True:
        one = cv2.imencode('.jpg', cam)[1].tobytes()
        yield (b'--one\r\n'b'Content-Type: image/jpeg\r\n\r\n' + one + b'\r\n')


def plate_vid():
    while True:
        two = cv2.imencode('.jpg', plate_cam)[1].tobytes()
        yield (b'--two\r\n'b'Content-Type: image/jpeg\r\n\r\n' + two + b'\r\n')


# this is our update route where we are going to update our employee
@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Data.query.get(request.form.get('id'))

        my_data.name = request.form['name']
        my_data.Model = request.form['Model']
        my_data.plate = request.form['plate']

        db.session.commit()
        flash("Vehicle License Details Updated Successfully")

        return redirect(url_for('Index'))


# This route is for deleting our employee
@app.route('/delete/<id>/', methods=['GET', 'POST'])
def delete(id):
    my_data = Data.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Vehicle License Deleted Successfully")

    return redirect(url_for('Index'))


@app.route('/video')
def video():
    return Response(live_vid(), mimetype='multipart/x-mixed-replace; boundary=one')


@app.route('/video2')
def video2():
    return Response(plate_vid(), mimetype='multipart/x-mixed-replace; boundary=two')


@app.route('/home')
def homepage():
    return render_template("home.html")


@app.route('/rand')
def rand():
    return render_template("rand.html")


@app.before_first_request
def before_first_request():
    threading.Thread(target=updating).start()


def updating():
    with app.app_context():
        while True:
            time.sleep(4)
            turbo.push(turbo.replace(render_template('rand.html'), 'fun'))


if __name__ == "__main__":
    app.run(debug=True, host="192.168.43.78")
    # use WLAN IP as host to run on local area network
    # use server IP as host to be accessible over the internet if hosted on cloud linux server
