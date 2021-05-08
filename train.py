import sys
import random
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import dev

PORT = 587 # gmail
JOB_ID = str(hash(random.uniform(0,1)))

def send_email(acc, rec, text, attachments=[]):
    global PORT,JOB_ID
    snd,pw = acc
    msg = MIMEMultipart()
    msg["From"] = snd
    msg["To"] = rec
    msg["Subject"] = f"job{JOB_ID[:10]}..."
    if attachments is not None:
        for att_path in attachments:
            att = MIMEBase('application',
                'octet-stream')
            with open(att_path,'rb') as att_bytes:
                att.set_payload(att_bytes.read())
            encoders.encode_base64(att)
            att.add_header("Content-Disposition",
                "attachment; filename= "+att_path)
            msg.attach(att)
    msg.attach(MIMEText(text + f"\n\nJOB_ID: {JOB_ID}\nsystem time: {time.time()}"))
    print(msg)
    with smtplib.SMTP('smtp.gmail.com', PORT) as sess:
        sess.starttls()
        print('tls started')
        sess.login(snd,pw)
        print('login successful')
        sess.sendmail(snd,rec,msg.as_string())
        print('message sent')

def job(epochs, minibatch_size, nb_augments):
    cifar10 = dev.MultiCIFAR10(nb_augments,
        root='data',
        download=True,
        train=True,
        transform=dev.train_transform)
    train_loader = dev.DataLoader(cifar10,
        batch_size = minibatch_size,
        shuffle=True)
    relenc = dev.RelationalEncoder()
    log = relenc.train(epochs, minibatch_size, nb_augments, train_loader)
    enc = relenc.encoder
    rel = relenc.relation_head
    dev.save_model(enc.state_dict(), "enc.tar")
    dev.save_model(rel.state_dict(), "rel.tar")
    return ["enc.tar", "rel.tar", log]

def main():
    epochs = int(sys.argv[1])
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    src_acc = sys.argv[4:6]
    rec_adr = sys.argv[6]
    send_email(src_acc, rec_adr, f"job started\nparams: {epochs} {m} {k}")
    result_files = job(epochs, m, k)
    send_email(src_acc, rec_adr, f"job complete\nparams: {epochs} {m} {k}",
        attachments = result_files)
    print(result_files)

if __name__ == "__main__":
    main()
