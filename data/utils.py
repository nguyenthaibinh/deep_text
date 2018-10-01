# import email
import re
import time
import numpy as np
from pprint import pprint


def parse_raw_message(raw_message):
    global n_attachment
    lines = raw_message.split('\n')
    email_dict = {}
    message = ''
    keys_to_extract = ['from', 'to', 'subject', 'date']
    for line in lines:
        email_dict['attachment'] = None
        if "File: " in line:
            try:
                m_f = re.findall(r'File:\s+(.*?\.\w+)', line)
                if len(m_f) > 0:
                    email_dict['attachment'] = m_f
                    # print("M_F:", m_f)
                    # print("M_F.type()", type(m_f))
                    n_attachment += 1
            except Exception as e:
                # print("LINE:", line)
                # print("EXCEPT:", e)
                pass
        elif ':' not in line:
            message += line.strip()
            email_dict['body'] = message
        else:
            line = re.sub('\s+', ' ', line)
            # print("LINE:", line)
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                # print("KEY:", key)
                email_dict[key] = val
                # print("EMAIL[{}]:".format(key), email[key])

    try:
        email_dict['body'] = re.sub('\s+', ' ', email_dict['body'])
    except:
        pass
    try:
        email_dict['subject'] = re.sub('\s+', ' ', email_dict['subject'])
    except Exception as e:
        print("SUBJECT_RAW_MESSAGE:", raw_message)
        # print("SUBJECT:", email['subject'])
        pass
    try:
        email_dict['to'] = re.sub('\s+', ' ', email_dict['to'])
    except Exception as e:
        # print("TO_RAW_MESSAGE:", raw_message)
        pass
    return email_dict


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }


def parse_message_attachment(raw_message):
    lines = raw_message.split('\n')
    attachments = []
    # print("LINES:", lines)
    for line in lines:
        if ("Subject:" in line) or ("To:" in line) or ("From:" in line) or ("mailto:" in line):
            continue
        if "File: " in line:
            # print("LINE:", line)
            try:
                m_f = re.findall(r'File:\s+(.*?\.\w+)', line)
                if len(m_f) > 0:
                    attachments += m_f
            except Exception as e:
                pass
    return attachments


def pasrse_corpus_attachments(messages):
    attachments = []
    for message in messages:
        att = parse_message_attachment(message)
        attachments.append(att)


def parse_message_id(raw_message):
    global n_attachment
    lines = raw_message.split('\n')
    mail_body = ''
    message_id = ''
    message = ''
    # keys_to_extract = ['message-id']
    for line in lines:
        if ('Message-ID:' not in line) and ('X-Folder' not in line):
            message += "{}\n".format(line)
            # re.sub('\s+',' ', line)
            mail_body = message
        elif ('Message-ID:' in line):
            line = re.sub('\s+', ' ', line)
            # print("LINE:", line)
            pairs = line.split(':')
            # key = pairs[0].lower()
            val = pairs[1].strip()
            message_id = val
    return mail_body, message_id


# Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs


def extract_single_messages(message_thread):
    """
    Input: a thread of messages
    Output: a list whose elements are single messages

    Below is an example of a message_thread:

    Message-ID:  <18712784.1075861051497.JavaMail.evans@thyme>
    Subject:     RE: draft letter for calculation of settlement amount/methodology

    Financial trading is just now gearing up to send these follow-up letters.  The few that have been sent have been incorporated into the counterparty files and logged onto the "Master Letter Log" which is updated by Stephanie Panus.

    Sara

     -----Original Message-----
    From:   Bruce, Robert
    Sent:   Monday, February 25, 2002 11:23 AM
    To: Shackleton, Sara
    Subject:    RE: draft letter for calculation of settlement amount/methodology

    thanks

    are we keeping a log of these as we send them out -- Bob

     -----Original Message-----
    From:   Shackleton, Sara
    Sent:   Monday, February 25, 2002 11:21 AM
    To: Bruce, Robert
    Subject:    draft letter for calculation of settlement amount/methodology

    Bob:

    The attached letter can be used for terminated or non-terminated parties; just tweak the letter.

     << File: #631941 v2 - Draft letter from Enron to XYZ Corp. - January 20021.doc >>

    Sara Shackleton
    Enron Wholesale Services
    1400 Smith Street, EB3801a
    Houston, TX  77002
    Ph:  (713) 853-5620
    Fax: (713) 646-3490
    #631941 v2 - Draft letter from Enron to XYZ Corp. - January 20021.doc
    """
    messages = []
    lines = message_thread.split('\n')
    message = ""

    reading_new_message = True

    for line in lines:
        if "-- Forwarded by" in line:
            reading_new_message = False
        if "Subject:" in line:
            reading_new_message = True
            continue
        elif "-Original Message-" in line:
            reading_new_message = False
            messages.append(message)
            message = ""
            continue
        elif "File: " in line:
            reading_new_message = False
            messages.append(message)
            message = ""
            continue
        elif "Regard: " in line:
            reading_new_message = False
            messages.append(message)
            message = ""
            continue
        elif "Sincerely: " in line:
            reading_new_message = False
            messages.append(message)
            message = ""
            continue
        if reading_new_message is True:
            # print("line:", line)
            if message == "":
                message += line.strip()
            else:
                message += " {}".format(line.strip())
    if reading_new_message is True:
        messages.append(message)

    return messages


def form_message_pairs(messages):
    """
    Input: a list of consecutive messages
    Output: a list of message pairs: (request, response)
    """
    n_messages = len(messages)
    message_pairs = []

    for i in range(n_messages - 1, 0, -1):
        try:
            request = messages[i]
            response = messages[i - 1]
            message_pairs.append((request, response))
        except Exception as e:
            print("exception:", str(e))
            print("i:", i)
    return message_pairs


def _write_elapsed_time(start_t=None, msg=None):
    cur_time = time.time()
    if start_t is None:
        print("{}".format(msg))
    else:
        elapsed_time = time.strftime("%H:%M:%S",
                                     time.gmtime(cur_time - start_t))
        print("{}".format(msg), elapsed_time)
    return cur_time


def negative_sampling(n_items, type="normal", repeat=True):
    neg = np.random.randint(n_items)
    return neg


def get_train_examples(train_requests, train_responses, n_items, n_neg,
                       verbose=False, weights=None):
    start_t = time.time()
    requests, responses, labels = [], [], []
    n_pairs = len(train_requests)

    if verbose:
        print("n_pairs:", n_pairs, "n_neg:", n_neg)
    for i in range(len(train_requests)):
        # positive instance
        requests.append(train_requests[i])
        responses.append(train_responses[i])
        labels.append(1)

        # negative instances
        for t in range(n_neg):
            j = negative_sampling(n_items=n_items)
            """
            try:
                while train.has_key((u, j)):
                    j = np.random.randint(n_items)
            except:
                while (u, j) in train.keys():
                    j = np.random.randint(n_items)
            """
            requests.append(train_requests[i])
            responses.append(j)
            labels.append(0)
    if verbose:
        _write_elapsed_time(start_t, "{} train instances obtained:".format(len(requests)))
    requests = np.array(requests, dtype=np.int64)
    responses = np.array(responses, dtype=np.int64)
    labels = np.array(labels, dtype=np.float32)
    return requests, responses, labels


def standardize_sentence(s):
    """
    - Remove all non alphabet, dot, colon, semi-colon of a given string.
    - Remove double spaces
    """
    ret = 0
    return ret


def write_vocab(vocab, file_name):
    f = open(file_name, "w")
    pprint(vocab._mapping, f)
    f.close()

if __name__ == '__main__':
    raw_message = """
    Message-ID:  <18712784.1075861051497.JavaMail.evans@thyme>
    Subject:     RE: draft letter for calculation of settlement amount/methodology

    Financial trading is just now gearing up to send these follow-up letters.  The few that have been sent have been incorporated into the counterparty files and logged onto the "Master Letter Log" which is updated by Stephanie Panus.

    Sara

     -----Original Message-----
    From:   Bruce, Robert
    Sent:   Monday, February 25, 2002 11:23 AM
    To: Shackleton, Sara
    Subject:    RE: draft letter for calculation of settlement amount/methodology

    thanks

    are we keeping a log of these as we send them out -- Bob

     -----Original Message-----
    From:   Shackleton, Sara
    Sent:   Monday, February 25, 2002 11:21 AM
    To: Bruce, Robert
    Subject:    draft letter for calculation of settlement amount/methodology

    Bob:

    The attached letter can be used for terminated or non-terminated parties; just tweak the letter.

     << File: #631941 v2 - Draft letter from Enron to XYZ Corp. - January 20021.doc >>

    Sara Shackleton
    Enron Wholesale Services
    1400 Smith Street, EB3801a
    Houston, TX  77002
    Ph:  (713) 853-5620
    Fax: (713) 646-3490
    #631941 v2 - Draft letter from Enron to XYZ Corp. - January 20021.doc
    """

    m2 = """
    Traveling to have a business meeting takes the fun out of the trip.  Especially if you have to prepare a presentation.  I would suggest holding the business plan meetings here then take a trip without any formal business meetings.  I would even try and get some honest opinions on whether a trip is even desired or necessary.

As far as the business meetings, I think it would be more productive to try and stimulate discussions across the different groups about what is working and what is not.  Too often the presenter speaks and the others are quiet just waiting for their turn.   The meetings might be better if held in a round table discussion format.

My suggestion for where to go is Austin.  Play golf and rent a ski boat and jet ski's.  Flying somewhere takes too much time.
    """
    # print("raw_message:\n", raw_message)
    messages = extract_single_messages(m2)
    print("messages:", messages)
    print("n_messages:", len(messages))

    pairs = form_message_pairs(messages)
    print("pairs:", pairs)
    print("n_pairs:", len(pairs))
