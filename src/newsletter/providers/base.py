from __future__ import annotations

from typing import Protocol


class EmailProvider(Protocol):
    def create_campaign(self, *, title: str, subject: str) -> str:
        ...

    def set_campaign_content(self, *, campaign_id: str, html: str, text: str) -> None:
        ...

    def send_test(self, *, campaign_id: str, test_email: str) -> None:
        ...

    def send_campaign(self, *, campaign_id: str) -> None:
        ...
