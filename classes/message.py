from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Dict

# BaseModel
class Message(BaseModel):
    role: str
    content: str

    @field_validator('role', 'content', mode='before')
    @classmethod
    def str_validator(cls, value):
        if not isinstance(value, str):
            raise TypeError('must be a string')
        return value

    def to_dict(self) -> dict:
        return self.dict()

external_data: Dict[str, str] = {
    "role": "user",
    "content": "Generate Lorem ipsum for me."
}

def main() -> None:
    # Example
    msg = Message(**external_data)
    print(msg)

# Main
if __name__ == '__main__':
    main()
