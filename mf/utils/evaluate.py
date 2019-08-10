'''
Rewritten in python based on Evaluate.java, originally
/**
 * Created by ghostof2007 on 5/8/14.
 * Different evaluation routines
 */
'''


def evaluate(gold_seg_file, pred_seg_file, quiet=False, debug=False):

    def get_seg_points(segmentation):
        seg_points = set()
        i = 0
        for c in segmentation:
            if c == '-': seg_points.add(i)
            else: i += 1
        return seg_points

    def evaluate_seg_points(pred_seg, gold_segs):
        # find the best match over different points
        best_correct, best_total, min_best_total = 0.0, 0.0, 100.0
        pred_points = get_seg_points(pred_seg)
        for gold_seg in gold_segs:
            gold_points = get_seg_points(gold_seg)
            gold_size = len(gold_points)
            correct = len(gold_points & pred_points)
            if correct > best_correct or correct == best_correct and gold_size < best_total:
                best_correct = correct
                best_total = gold_size
            if gold_size < min_best_total:
                min_best_total = gold_size
        if best_total == 0:
            best_total = min_best_total
        return (best_correct, best_total, len(pred_points))

    def print_segs(segs):
        for word, seg in segs.iteritems():
            print(word + ' # ' + seg)

    predicted_segs, incorrect_segs, correct_segs = dict(), dict(), dict()
    correct, pred_total, gold_total = 0.0, 0.0, 0.0

    gold_segs = dict()
    # print(0)
    with open(gold_seg_file, 'r', encoding='utf8') as fin:
        for line in fin:
            # line = line.encode('utf8')
            # if line[:5] == 'piirr': print(list(line))
            segs = line.strip().split(':')
            assert len(segs) % 2 == 0, segs
            segs = ':'.join(segs[: len(segs) // 2]), ':'.join(segs[len(segs) // 2:])
            gold_segs[segs[0]] = segs[1].split()
    pred_segs = dict()
    with open(pred_seg_file, 'r', encoding='utf8') as fin:
        try:
            for line in fin:
                # if line[:5] == 'piirr': print(list(line))
                segs = line.strip().split(':')
                assert len(segs) % 2 == 0, segs
                segs = ':'.join(segs[: len(segs) // 2]), ':'.join(segs[len(segs) // 2:])
                pred_segs[segs[0]] = segs[1]      # assert only one prediction, no alternatives
        except:
            print(line)
            import pdb; pdb.set_trace()

    for word in gold_segs:
        pred_seg = pred_segs[word]
        segs = gold_segs[word]
        res = evaluate_seg_points(pred_seg, segs)
        correct += res[0]
        gold_total += res[1]
        pred_total += res[2]
        predicted_segs[word] = pred_seg
        if res[2] != res[0]:
            incorrect_segs[word] = pred_seg + ' : ' + str(segs)
        else:
            correct_segs[word] = pred_seg + ' : ' + str(segs)
        if res[2] < res[1] and debug:
            print(pred_seg, segs)
            raw_input()

    if not quiet:
        print("Incorrect segmentations:")
        print_segs(incorrect_segs)
        print("\nCorrect segmentations:")
        print_segs(correct_segs)
        print("\nAll segmentations:")
        print_segs(pred_segs)
    p = correct / pred_total
    r = correct / gold_total
    f = 2 * p * r / (p + r)
    print("Correct: %s\tGoldTotal: %s\tPredTotal: %s" % (correct, gold_total, pred_total))
    print("Precision: %s\tRecall: %s\tF1: %s" % (p, r, f))
    return (p, r, f)
