async def test_admin_pages_reachable_and_share_nav(client):
    for path in ("/kb", "/admin"):
        r = await client.get(path)
        assert r.status_code == 200
        assert "/static/admin.js" in r.text
