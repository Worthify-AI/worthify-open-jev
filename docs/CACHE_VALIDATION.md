# Cache validation

Cached scoring is disabled in Worthify release code. On the owned 37-state × 21-decision Qwen shape777 fixture (777 decisions), serial prefix caching differed from fresh direct scoring on 5 argmax decisions (maximum probability error 0.0934). Parallel shared-prefix caching differed on 6 decisions (maximum probability error 0.0624).

These measurements apply to the pinned upstream implementation at commit `53e3028363509f8533d90fe82d983770da1f6c02` and the reported Qwen reproduction. They do not establish cache equivalence for any model, quantization, or future runtime. `direct` is therefore the only Worthify release scoring mode.

The original experimental implementation remains in the repository for inspection. To reproduce its upstream experiment, use a separate checkout pinned to that commit and run `benchmarks/shape777.py` there with its documented model revision and one visible GPU. The release CLI intentionally provides no cache-override flag.
