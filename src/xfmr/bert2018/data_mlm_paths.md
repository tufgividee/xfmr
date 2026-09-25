                Tokenized Dataset
                       │
                       ▼
                 Construct A/B
                       │
                       ▼
                  Apply MLM
                       │
                masked examples
                       │
             ┌─────────┴─────────┐
             │                   │
          STATIC              DYNAMIC
             │                   │
          save disk          use directly
             │                   │
  REUSE - load disk              │
             │                   │
             └─────────┬─────────┘
                       ▼
                   DataLoader
                       │
        colllate / pad / batch / tensor
                       │
                       ▼
                      BERT