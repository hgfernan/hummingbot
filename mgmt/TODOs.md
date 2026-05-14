# TODOs

## Overvu  table

| **Sequence** | **Priority** | **Status** | **Requisites** | **Description**                                       |
| -----------: | :----------: | :--------: | :------------- | :---------------------------------------------------- |
|            1 |            2 |  INACTIVE  |                | Recompile Hummingbot without `gateway`                |
|            2 |            1 |  INACTIVE  |                | Get updated version from upstream                     |
|            3 |            1 |  INACTIVE  | 2,             | Apply current changes to upstream version             |
|            4 |            0 |  INACTIVE  |                | Restart development of recursive scalper              |
|            5 |            0 |  INACTIVE  |                | Create a booklet to register Hummingbot routine       |
|            6 |            0 |  INACTIVE  |                | Have a running main `brsc` class with mocked children |
|            7 |            0 |  INACTIVE  |                | Have a running main `brsc` class with base child      |
|            8 |            0 |  INACTIVE  |                | Have a running main `brsc` class with quote child     |
|            9 |            0 |  INACTIVE  |                | Have DeepSeek handout at hand                         |
|           10 |            1 |  INACTIVE  | 3,             | Add paper exchanges to configuration                  |
|           11 |            1 |  INACTIVE  | 0,5,           | Run some existent strategies                          |
|           12 |            1 |  INACTIVE  | 11,            | Study the code of the strategies that were run        |
|           13 |            1 |  INACTIVE  | 0,5,           | Run some existent scripts                             |
|           14 |            1 |  INACTIVE  | 13,            | Study the code of the scripts that were run           |
|           15 |            0 |  DONE      |                | Enable a new password                                 |
|           16 |            1 |  INACTIVE  |                | Learn how to use Binance price oracle                 |
|           17 |            2 |  INACTIVE  |                | Learn how to check the current Hummingbot version     |
|           18 |            2 |  INACTIVE  |                | Make sure forked code has the same Hummingbot version |
|           19 |            1 |  INACTIVE  |                | Study the new timing algo suggested by Gemini         |
|           20 |            2 |  INACTIVE  |                | Enable Hummingbot testing in vscode                   |
|           21 |            2 |  INACTIVE  |                | Apply mocked unit testing for the new classes         |

## Day by day

### 2026-05-14

1. Change password;

2. Start working in the recursive scalper implementation

### 2026-03-09

1. Make sure the initialization of the helpers in `RoundTripTrading` is correct;

2. Confirm the helper will transit from `RttState.START` to `RttState.TRANSFORM_CALC`;

3. Confirm `RttState.TRANSFORM_CALC` will provide buy order parameters correctly;

4. Eliminate remains of previous `download_order_book_and_trades.py`, as the price
    estimation currently is simpler;
