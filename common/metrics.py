from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, precision_score, recall_score, f1_score

def metrics(labels, logits)
    auc = roc_auc_score(labels, logits)
    loss = log_loss(labels, logits)
    acc = accuracy_score(labels, logits.round())
    precision = precision_score(labels, logits.round())
    recall = recall_score(labels, logits.round())
    f1_score = f1_score(labels, logits.round())
    return auc, loss, acc, precison, recall, f1_score
