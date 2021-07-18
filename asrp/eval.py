import jiwer
import editdistance as ed


def cer(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot


def wer(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.lower().split(' ')
        t = t.lower().split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


def chunked_cer(targets, predictions, chunk_size=None):
    _predictions = [char for seq in predictions for char in list(seq)]
    _targets = [char for seq in targets for char in list(seq)]
    if chunk_size is None: return jiwer.wer(_targets, _predictions)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(targets):
        _predictions = [char for seq in predictions[start:end] for char in list(seq)]
        _targets = [char for seq in targets[start:end] for char in list(seq)]
        chunk_metrics = jiwer.compute_measures(_targets, _predictions)
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
    return float(S + D + I) / float(H + S + D)


def chunked_wer(targets, predictions, chunk_size=None):
    if chunk_size is None: return jiwer.wer(targets, predictions)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(targets):
        chunk_metrics = jiwer.compute_measures(targets[start:end], predictions[start:end])
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
    return float(S + D + I) / float(H + S + D)
