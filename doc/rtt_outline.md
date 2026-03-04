# Outline of Round Trip Trading in the Hummingbot context

## Introduction

## Quote accumulation outline

```Python
class RoundTripTrading(ScriptStrategyBase):

    # TODO to be used class has an associated config class
    @classmethod
    def init_markets(cls, config: BaseModel):
        pass

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        pass

    def on_tick(self):
        pass
```
