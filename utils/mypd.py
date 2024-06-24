# Internal Modules
...
# External Modules
import pandas as pd
import logging

# Root 
logger_name = 'utils.mypd'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
# File Handler
file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)


class MyDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def replace_semicolon_with_comma(self) -> pd.DataFrame:
        # 모든 셀에 대해 문자열 변환을 시도합니다.
        return self.applymap(lambda x: str(x).replace(';', ',') if pd.notnull(x) else x)

def main() -> None:
    ...

# Main
if __name__ == '__main__':
    main()
