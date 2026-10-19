/* 后台管理外壳:一份共用导航挂在后台各页上(知识库录入 / 后台首页)。
 *
 * 各页样式各自内联,所以导航自带样式、自己注入,只往 <head> 追加一个 <style id="mh-nav-css">,
 * 不碰宿主页的 CSS;整个文件包在 IIFE 里,只往 window 上挂一个 mountAdminNav,
 * 免得和宿主页自己的 $ / el 之类的短名字撞名。
 *
 * 模块入口就是各章原本的路径,导航只把它们收到一处,不做跳转改写——文档里贴的链接照样能用。
 * ch03 只有「知识库」一个模块,模块下的子页行放的是录入页六个区块的页内锚点(#manual …),
 * 后面加模块只需往 NAV 里补一行。 */
(function () {
  const NAV = [
    { href: "/admin", label: "后台首页" },
    {
      href: "/kb", label: "知识库",
      children: [
        ["/kb#manual", "① 手工录入"],
        ["/kb#materials", "② 建库材料"],
        ["/kb#mining", "③ 对话挖知识"],
        ["/kb#dualwrite", "④ 向量化与双写"],
        ["/kb#search", "⑤ 检索自测"],
        ["/kb#recent", "⑥ 最近入库"],
      ],
    },
    { href: "/rag-eval", label: "RAG 评估" },
  ];

  const CSS = `
  .mh-nav { margin-top: 12px; font-size: 13px; }
  .mh-nav .row {
    display: flex; flex-wrap: wrap; align-items: stretch;
    background: #fff; border: 1px solid #dee2e6; border-radius: 10px;
    overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  .mh-nav .row .brand {
    display: flex; align-items: center; padding: 8px 13px; font-size: 12px;
    background: #4c6ef5; color: #fff; letter-spacing: 1px; font-weight: 600;
  }
  .mh-nav .row a {
    display: flex; align-items: center; padding: 8px 14px; text-decoration: none;
    color: #343a40; border-right: 1px solid #e9ecef; font-weight: 500;
  }
  .mh-nav .row a:hover { background: #f1f3f5; }
  .mh-nav .row a.on { background: #edf2ff; color: #3b5bdb; font-weight: 700; }
  .mh-nav .row .grow { flex: 1; border-right: 0; }
  .mh-nav .row a.chat { color: #868e96; font-weight: 400; }
  .mh-nav .sub {
    display: flex; flex-wrap: wrap; align-items: center; gap: 2px;
    border: 1px solid #dee2e6; border-top: 0; border-radius: 0 0 10px 10px;
    background: #f8f9fa; padding: 3px 6px;
  }
  .mh-nav .sub a {
    padding: 4px 10px; font-size: 12px; text-decoration: none;
    color: #495057; border-radius: 6px;
  }
  .mh-nav .sub a:hover { background: #e9ecef; }
  .mh-nav .sub a.on { background: #edf2ff; color: #3b5bdb; font-weight: 700; }
  `;

  function injectCss() {
    if (document.getElementById("mh-nav-css")) return;
    const s = document.createElement("style");
    s.id = "mh-nav-css";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function link(href, label, cls) {
    const a = document.createElement("a");
    a.href = href;
    if (cls) a.className = cls;
    a.textContent = label;
    return a;
  }

  /** 当前页归属哪个模块:精确命中优先,其次按前缀(/kb#manual 归 /kb)。 */
  function moduleOf(active) {
    const path = String(active || "").split("#")[0];
    return NAV.find((m) => m.href === path)
      || NAV.find((m) => m.href !== "/" && path.startsWith(m.href + "/"));
  }

  /** 渲染导航(不挂载):模块行 + 当前模块的子页行;挂不挂、挂哪儿交给 mountAdminNav。 */
  function renderAdminNav(active) {
    injectCss();
    const wrap = document.createElement("div");
    wrap.className = "mh-nav";

    const row = document.createElement("div");
    row.className = "row";
    const brand = document.createElement("span");
    brand.className = "brand";
    brand.textContent = "后台管理";
    row.appendChild(brand);
    const mod = moduleOf(active);
    for (const m of NAV) row.appendChild(link(m.href, m.label, m === mod ? "on" : ""));
    const grow = document.createElement("span");
    grow.className = "grow";
    row.appendChild(grow);
    row.appendChild(link("/", "聊天页 →", "chat"));
    wrap.appendChild(row);

    if (mod && mod.children) {
      const sub = document.createElement("div");
      sub.className = "sub";
      for (const [href, label] of mod.children) sub.appendChild(link(href, label, ""));
      wrap.appendChild(sub);
    }
    return wrap;
  }

  /** 挂到顶栏底下(顶栏是每页自己的标题条);页上没有 .topbar 就摆在 .wrap 最前面。 */
  function mountAdminNav(active) {
    const nav = renderAdminNav(active);
    const bar = document.querySelector(".topbar");
    if (bar && bar.parentNode) bar.parentNode.insertBefore(nav, bar.nextSibling);
    else {
      const w = document.querySelector(".wrap") || document.body;
      w.insertBefore(nav, w.firstChild);
    }
    return nav;
  }

  window.mountAdminNav = mountAdminNav;
})();
