from __future__ import annotations

import requests

from newsletter.providers.base import EmailProvider


class MailchimpProvider(EmailProvider):
    def __init__(
        self,
        *,
        api_key: str,
        server_prefix: str,
        audience_id: str,
        from_name: str,
        reply_to: str,
    ) -> None:
        self.base_url = f"https://{server_prefix}.api.mailchimp.com/3.0"
        self.auth = ("newsletter", api_key)
        self.audience_id = audience_id
        self.from_name = from_name
        self.reply_to = reply_to

    def create_campaign(self, *, title: str, subject: str) -> str:
        payload = {
            "type": "regular",
            "recipients": {"list_id": self.audience_id},
            "settings": {
                "subject_line": subject,
                "title": title,
                "from_name": self.from_name,
                "reply_to": self.reply_to,
            },
        }
        response = self._request("POST", "/campaigns", json=payload)
        campaign_id = response.get("id")
        if not campaign_id:
            raise RuntimeError("Mailchimp did not return a campaign ID.")
        return str(campaign_id)

    def set_campaign_content(self, *, campaign_id: str, html: str, text: str) -> None:
        self._request(
            "PUT",
            f"/campaigns/{campaign_id}/content",
            json={"html": html, "plain_text": text},
        )

    def send_test(self, *, campaign_id: str, test_email: str) -> None:
        self._request(
            "POST",
            f"/campaigns/{campaign_id}/actions/test",
            json={"test_emails": [test_email], "send_type": "html"},
            expect_json=False,
        )

    def send_campaign(self, *, campaign_id: str) -> None:
        self._request("POST", f"/campaigns/{campaign_id}/actions/send", expect_json=False)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        expect_json: bool = True,
    ) -> dict:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            auth=self.auth,
            json=json,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Mailchimp API error {response.status_code}: {response.text}")
        if not expect_json or not response.text:
            return {}
        return response.json()
