import sys
import getopt
import RunCQ_Basic
import RunCQ_Fast
import RunCR_DBLP


if __name__ == '__main__':

    algorithm = "cq_basic"
    alpha = 0.2
    c = 0.85
    epsilon = 0.003
    max_iter = 1000
    query_node = 121
    source = 0
    target = 19
    k = 10
    dataset = "../data/DBLP_NoN.npy"

    opts, args = getopt.getopt(sys.argv[1:], "h", ["algorithm=", "alpha=", "c=", "query_node=", "source=", "target=", "k=", "max_iter=", "epsilon=", "dataset="])
    for option, value in opts:
        if option == "-h":
            print "Welcome, this is a program of NoN Cross Query"
            print ""
            print "python __init__.py [option] [argument]"
            print ""
            print "Option and arguments:"
            print "--algorithm      The algorithm used to do the query."
            print "                 cq_basic:   requires --alpha --c --query_node --source --target --k --dataset"
            print "                 cq_fast:    requires --alpha --c --epsilon --query_node --source -- target --k --dataset"
            print "                 cr:         requires --alpha --c --max_iter --epsilon --query_node --source -- target --dataset"
            print "--alpha          The regularization parameter for cross-network consistency."
            print "--c              The regularization parameter for query preference."
            print "--max_iter       The maximal number of iteration for updating ranking vector."
            print "--epsilon        In cq_fast, epsilon is the error factor to control the accuracy of results; In cr, epsilon is the convergence parameter."
            print "--query_node     The ID of the query node of interest."
            print "--source         The ID of the source domain-specific network."
            print "--target         The ID of the target domain-specific network."
            print "--k              The number of retrieved nodes."
            print "--dataset        The path for the dataset."
            print ""
            print "Example:"
            print ""
            print "python __init__.py --algorithm cq_basic --alpha 0.2 --c 0.85 --query_node 121 --source 0 --target 19 --k 10 --dataset ../data/DBLP_NoN.npy"
            print ""
            print "python __init__.py --algorithm cq_fast --alpha 0.2 --c 0.85 --epsilon 0.003 --query_node 121 --source 0 --target 19 --k 10 --dataset ../data/DBLP_NoN.npy"
            print ""
            print "python __init__.py --algorithm cr --alpha 0.2 --c 0.85 --max_iter 1000 --epsilon 1e-15 --query_node 121 --source 0 --target 19 --k 10 --dataset ../data/DBLP_NoN.npy"
            print ""
            exit(0)
        if option == "--algorithm":
            algorithm = value
        if option == "--alpha":
            alpha = float(value)
        if option == "--c":
            c = float(value)
        if option == "--max_iter":
            max_iter = int(value)
        if option == "epsilon":
            epsilon = float(value)
        if option == "--query_node":
            query_node = int(value)
        if option == "--source":
            source = int(value)
        if option == "--target":
            target = int(value)
        if option == "--k":
            k = int(value)
        if option == "--dataset":
            dataset = value

    if algorithm == "cq_basic":
        print "------- CQ_Basic -------"
        TopKAuthorNames = RunCQ_Basic.run_cq_basic(alpha, c, query_node, source, target, k, dataset)
        print "\nTop K Author Names:"
        for author in TopKAuthorNames:
            print author[0]
        print "------------------------"
    elif algorithm == "cq_fast":
        print "------- CQ_Fast --------"
        [TopKAuthorNames_fast, RelevantDomains] = RunCQ_Fast.run_cq_fast(alpha, c, epsilon, query_node, source, target, k, dataset)
        print "\nTop K Author Names:"
        for author in TopKAuthorNames_fast:
            print author[0]
        # print ""
        # print "Relevant Domains:"
        # for domain in RelevantDomains:
        #     print domain[0]
        print "------------------------"
    elif algorithm == "cr":
        print "---------- CR ----------"
        TopKAuthorNames_CR = RunCR_DBLP.run_cr_dblp(alpha, c, max_iter, epsilon, query_node, source, target, k, dataset)
        print "\nTop K Author Names:"
        for author in TopKAuthorNames_CR:
            print author[0]
        print "------------------------"
    else:
        print "Invalid algorithm. Please try again!"
        exit(0)
