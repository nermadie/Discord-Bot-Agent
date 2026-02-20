def _extract_image_urls_from_attachments(attachments):
    """Extract image URLs from Discord attachment list."""
    image_urls = []
    for attachment in attachments or []:
        content_type = (attachment.content_type or "").lower()
        filename = (attachment.filename or "").lower()
        if content_type.startswith("image/") or filename.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        ):
            image_urls.append(attachment.url)
    return image_urls


def _attachment_context_for_summary(message):
    """Build compact attachment context appended to summary message rows."""
    attachments = list(message.attachments or [])
    if not attachments:
        return ""

    image_urls = _extract_image_urls_from_attachments(attachments)
    file_names = [a.filename for a in attachments if a.filename]

    parts = []
    if image_urls:
        parts.append("Ảnh: " + ", ".join(image_urls[:3]))
    if file_names:
        parts.append("File: " + ", ".join(file_names[:3]))

    return " | " + " | ".join(parts) if parts else ""
