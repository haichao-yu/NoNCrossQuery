import numpy as np
import RunCQ_Basic
import RunCQ_Fast


print "------- CQ_Basic -------"
TopKAuthorNames = RunCQ_Basic.run_cq_basic()
print "\nTop K Author Names:"
for author in TopKAuthorNames:
    print author[0]
print "------------------------\n\n"

print "------- CQ_Fast --------"
[TopKAuthorNames_fast, RelevantDomains] = RunCQ_Fast.run_cq_fast()
print "\nTop K Author Names:"
for author in TopKAuthorNames:
    print author[0]
print ""
print "Relevant Domains:"
for domain in RelevantDomains:
    print domain[0]
print "------------------------\n\n"
