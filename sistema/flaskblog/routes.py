import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort, Flask, request, render_template, send_from_directory
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import (RegistrationForm, LoginForm, UpdateAccountForm,
                             PostForm, RequestResetForm, ResetPasswordForm, PatientForm)
from flaskblog.models import User, Post, Patient
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message

from flaskblog.defs import *
from flaskblog.def_plot import *
from flaskblog.cnn import *
import cv2
from scipy.ndimage import rotate
from scipy.misc import imread, imshow
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import cv2
import imutils
from operator import is_not
from functools import partial
from pylab import *
from random import *
from sklearn.cluster import KMeans
from matplotlib import transforms
import scipy.ndimage.morphology as morp
from skimage import feature



@app.route("/")
@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/ejemplo")
def ejemplo():
    return render_template('ejemplo.html', title='ejemplo')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    #picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)
    picture_path = os.path.join(app.root_path, 'flaskblog/static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='New Post',
                           form=form, legend='New Post')


@app.route("/patient/new", methods=['GET', 'POST'])
@login_required

def new_patient():
    form = PatientForm()
    if form.validate_on_submit():
        patient = Patient(name_patient=form.name_patient.data, observation=form.observation.data ,doctor=current_user)
        db.session.add(patient)
        db.session.commit()
        flash('Your patitent has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_patient.html', title='New Patient' ,form=form, legend='New Patient')


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)

@app.route("/patient/<int:patient_id>")
def patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('patient.html', name_patient=patient.name_patient, patient=patient)

@app.route("/patient_wall")
def patient_wall():
    '''
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    '''
    page = request.args.get('page', 1, type=int)
    patients = Patient.query.order_by(Patient.id).paginate(page=page, per_page=1000)
    return render_template('patient_wall.html', patients=patients)
   

'''
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)

'''



@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')


@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))


@app.route("/user/<string:username>")
def user_posts(username):
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(author=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page=page, per_page=5)
    return render_template('user_posts.html', posts=posts, user=user)


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)


@app.route("/diagnostic")
def diagnostic():
    return render_template('diagnostic.html', title='About')




    '''
    ____________________________________________________________________________________________________________________________________
    
'''


APP_ROOT = os.path.dirname(os.path.abspath(__file__))



@app.route("/upload", methods=["POST"])
def upload():

    target = os.path.join(APP_ROOT, 'static')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print("Save it to:", destination)
        file.save(destination)

        ruta = 'flaskblog/static/'+filename
        img = cv2.imread(ruta, cv2.IMREAD_COLOR)
        imagen2=img
        width = img.shape[1]
        print(width)

    print(filename)
    '''
    tipo=cnn(filename)
    print(tipo)
    if (tipo=='s'):
        division=12
    else:
        division=8
    print(division)
    '''
    division=12
    height = img.shape[0]
    width = img.shape[1]
    height_original=int(height)
    width_original=int(width)
    print("ANCHOOOOO: ",width_original)
    print("ALTOOOOOO: ",height_original)
    qua = int(width/10)
    qua2 = int(qua*3)
    qua7 = int(qua*7)
    img[0:height, 0:qua2] = [0]
    img[0:height, qua7:width] = [0]

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 198, 300, cv2.THRESH_BINARY)[1]
    edges2 = feature.canny(thresh, sigma=3)
    skel = skeletonize(edges2)

    extracto = nombre_archivo(filename)


    nuevo_nombre = extracto+'_gts.png'
    path = 'flaskblog/static/'

    cv2.imwrite(os.path.join(path,nuevo_nombre), skel.astype(np.uint8)*255)


    extracto = nombre_archivo(filename)
    complemento = '_gts.png'
    complemento_segundodebug='_puntos.png'
    titulo_final = extracto+complemento
    primer_debug=titulo_final
    segundo_debug= extracto+complemento_segundodebug
    print("RUTA DE PRIMER DEBUG",primer_debug)
    print("RUTA DE PRIMER DEBUG",segundo_debug)
    path_segundo_debug='flaskblog/static/'+segundo_debug

    img = cv2.imread(os.path.join(path,titulo_final))
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    ancho = int(width)
    altura = int(height)
    alfa = int(altura/division)
    cons = 0
    a = np.empty(((division+1), 50), dtype=object)

    for i in range(0, (division+1)):

        cons1 = cons
        cons2 = cons1+alfa
        image = img[cons1:cons2, 0:ancho]

        #convirtiendo a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #aplicando desenfoque gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #threshold?
        thresh = cv2.threshold(blurred, 60, 200, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        count = 1
        for c in cnts:
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                    cX, cY = 0, 0
            xx = str(cX)+","+str(cY)
            a[i][count] = [cX, cY]

            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
            #CIRCULO DE CENTRO
            cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
            #COORDENADAS
            cv2.putText(image, xx, (cX - 20, cY - 20),
                        #TIPO DE LETRA, COLOR?
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.imshow("Image", img)
            count = count+1
        
        cons = cons2
        i = i+1
    cv2.imwrite(path_segundo_debug, img)  

    cero = [0, 0]
    #puntos=plot_all(c,primer_debug)
    #print("URL DE PUNTOS", puntos)
    b = [[cero if x is None else x for x in c] for c in a]
    #b = b[::-1]
    #print("ESTO ES LA LISTA B:",b)
    
    b = igualador(b,division,alfa)
    #print("ESTO ES LA LISTA B REGULADA:",b)
    #b = reemplazador(b, division)
    
    ax = np.zeros(shape=((division+1), 1), dtype=object)
    contador = 0

    lis_2 = []
    

    for i in range(0, (division+1)):
        a = limpio(b[i])
        lis_2.append(a)

   
    lis_3 = []
    lis_3 = rellenador(lis_2, division, ancho, altura)
    print("LISTA 3333333333333333333333333333333", lis_3)

    #plot_all(lis_3)

    ax = seleccionador_kmeans(lis_3)
    
    print("LISTAAAAAAAAAAAA: ", ax)

    ancho2 = int(ancho/2)
    axx = np.asarray(ax)
    print("LISTAAAAAAAAAAAAXXXXXXX: ", axx)
    bx = np.array([[ancho2, altura]])
    cx = np.concatenate((axx, bx), axis=0)
    dx = np.array([[ancho2, 0]])

    if (ax[0][1] < alfa):
        axx = np.concatenate((dx, cx), axis=0)
        axx = np.delete(axx, 1, 0)
        axx = np.array(axx.T)
    else:
        axx = np.array(cx.T)

    dim = (ancho, altura)
    resized = cv2.resize(imagen2, dim, interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots()
    
    l_x = axx[0]
    l_x = l_x.tolist()
    l_y = axx[1]
    l_y = l_y.tolist()

    #de aqui para abajo es todo girado
    y = axx[0]
    x = axx[1]



    #print("___"*20)
    pi1, pi2 = punto_inflexion(x, y)
    #print("+++"*20)
    a, b = max_min(x, y)
    a_2 = a[1:3]
    b_2 = b[1:3]
    #print(a)
    #print("___"*20)
    #print(b)


    z = np.polyfit(x, y, 5)
    f = np.poly1d(z)

    #print("Ecuacion bonita: ")
    #print(f)
    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    x_new2 = x_new[::-1]

    y_new = f(x_new2)
    y_new = y_new[::-1]


    pre=img_plot(x_new2,y_new,filename,fig,ax,a_2,b_2)
    plot_rotate(pre,ax,width_original,height_original)

    #print(plot_all(lis_3,filename,ax))
    '''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
    

    extracto2= nombre_archivo(filename)
    extracto21=extracto2+'pre'
    complemento = '_gts.png'
    titulo_final2 = extracto21+complemento
    print(titulo_final2)
    return render_template("complete.html", image_original=filename, image_name=titulo_final2, image_filter=primer_debug, image_points=segundo_debug)


