import sys
import os
import random
import time
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import dev
import load_models

PORT = 587 # only works for gmail smtp
JOB_ID = str(hash(random.uniform(0,1))) # a (probably) unique id
RECORD_DEST = "" # parent directory in which a folder "job{JOB_ID}" is created, which stores the job records.
NOTIFY = False # relates to a condition in one of 3 states:
# if False, this script only outputs encoder weights, relation head weights, and the session record.
# if True with email params, the script will (in addition to above) send an email from a gmail account upon job start + end, the latter containing the results.
# if True without email params, the script will (in addition to above) print start, end, and results to the standard output.

def send_email(acc, rec, text, attachments=[]):
    global PORT, JOB_ID
    snd,pw = acc
    # build email content
    msg = MIMEMultipart()
    msg["From"] = snd
    msg["To"] = rec
    msg["Subject"] = f"job{JOB_ID}"
    # load and attach paths from attachments
    if attachments is not None:
        for att_path in attachments:
            att = MIMEBase('application',
                'octet-stream')
            with open(att_path,'rb') as att_bytes:
                att.set_payload(att_bytes.read())
            encoders.encode_base64(att)
            att_filename = att_path[att_path.rfind('/')+1:]
            att.add_header("Content-Disposition",
                "attachment; filename= "+att_filename)
            msg.attach(att)
    # appendix, for now just system time.
    msg.attach(MIMEText(text + f"\n\nsystem time: {time.time()}"))
    # send it!
    with smtplib.SMTP('smtp.gmail.com', PORT) as sess:
        sess.starttls()
        print('tls started')
        sess.login(snd,pw)
        print('login successful')
        sess.sendmail(snd,rec,msg.as_string())
        print('message sent')

def job(epochs, minibatch_size, nb_augments, enc_path = None, rel_path = None, rel_class = 0):
    global JOB_ID, RECORD_DEST, NOTIFY
    # init data, data loader, and models
    cifar10 = dev.MultiCIFAR10(nb_augments,
        root='data',
        download=True,
        train=True,
        transform=dev.train_transform)
    train_loader = dev.DataLoader(cifar10,
        batch_size = minibatch_size,
        shuffle=True)
    enc = None
    rel = None
    if enc_path is not None:
        enc = load_models.load_encoder(enc_path)
        if NOTIFY:
            print("loaded encoder from", enc_path)
    if rel_path is not None:
        rel = load_models.load_relation_head(rel_path, rel_class)
        if NOTIFY:
            print("loaded relation head from", rel_path)
    else:
        rel = dev.relation_head(rel_class)()
    relenc = dev.RelationalEncoder(encoder=enc,relation_head=rel)
    # train and unpack models
    record = relenc.train(epochs, minibatch_size, nb_augments, train_loader, verbose=NOTIFY)
    enc = relenc.encoder
    rel = relenc.relation_head
    # output results
    job_dir = os.path.join(RECORD_DEST, f"job{JOB_ID}")
    os.mkdir(job_dir)
    enc_path = os.path.join(job_dir, "enc.tar")
    rel_path = os.path.join(job_dir, "rel.tar")
    dev.save_model(enc.state_dict(), enc_path)
    dev.save_model(rel.state_dict(), rel_path)
    records_path = os.path.join(job_dir, "session_data.json")
    with open(records_path, 'w') as records_file:
        json.dump(record, records_file)
    return [enc_path, rel_path, records_path]

def main():
    global PORT, RECORD_DEST, JOB_ID, NOTIFY
    # configure
    epochs = int(sys.argv[1])
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    rel = int(sys.argv[4])
    if '-n' in sys.argv:
        i = sys.argv.index('-n')
        NOTIFY = True
        if len(sys.argv[1:]) > i:
            rec_adr = sys.argv[i+1]
            src_acc = sys.argv[i+2:i+4]
    if '-p' in sys.argv:
        i = sys.argv.index('-p')
        PORT = int(sys.argv[i+1])
    if '-d' in sys.argv:
        i = sys.argv.index('-d')
        RECORD_DEST = sys.argv[i+1]
    if '-enc' in sys.argv:
        i = sys.argv.index('-enc')
        enc_path = sys.argv[i+1]
    else:
        enc_path = None
    if '-rel' in sys.argv:
        i = sys.argv.index('-rel')
        rel_path = sys.argv[i+1]
    else:
        rel_path = None
    # job start
    if NOTIFY:
        text = f"job started\nparams: {epochs} {m} {k} {rel} \ndevice: {dev.device}"
        try:
            send_email(src_acc, rec_adr, text)
        except:
            print(text)
    result = job(epochs, m, k, enc_path, rel_path, rel_class = rel)
    # job end
    if NOTIFY:
        text = f"job complete\nparams: {epochs} {m} {k} {rel}"
        try:
            send_email(src_acc, rec_adr, text, attachments = result)
        except:
            print(text)
            print(result)

if __name__ == "__main__":
    main()
