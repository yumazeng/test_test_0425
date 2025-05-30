# 技術需求：JustHome - 首頁

## 1. 總覽

本文件概述了開發 JustHome 首頁的技術需求，如所提供圖片所示。此頁面作為使用者搜尋房產、查看特色房源並導航至網站其他部分的入口點。

## 2. 通用需求

*   **響應式設計：** 版面配置必須完全響應式，並能無縫適應各種螢幕尺寸（桌上型電腦、平板電腦、行動裝置）。
*   **跨瀏覽器相容性：** 頁面應在現代網頁瀏覽器（Chrome、Firefox、Safari、Edge）上正確呈現並保持功能一致。
*   **效能：**
    *   優化網頁傳輸圖片，確保快速載入時間。
    *   延遲載入首屏下方圖片 (Lazy load)（例如：房產卡片圖片）。
    *   最小化 HTTP 請求。
    *   高效的 JavaScript 和 CSS。
*   **無障礙性 (A11y)：**
    *   在適當之處實作 WAI-ARIA 屬性。
    *   確保所有互動元素皆可透過鍵盤導覽。
    *   為文字和使用者介面元素提供足夠的色彩對比度。
    *   為所有具意義的圖片提供 alt 文字。
*   **搜尋引擎優化 (SEO) 友善：** 使用語意化 HTML5 標記。確保頁面標題、meta 描述和標題結構適當。

## 3. 組件細分

### 3.1. 頁首 / 導覽列

*   **容器：**
    *   深色背景。
    *   全寬、固定或滾動時頂部吸附 (sticky)。
*   **Logo：**
    *   元素：圖片 (`<img>` 或 SVG)。
    *   來源：`JustHome` Logo。
    *   連結：導航至首頁。
*   **導覽連結：**
    *   元素：連結列表 (`<ul>`, `<li>`, `<a>`)。
    *   項目："Home"、"Listings"、"Members"、"Blog"、"Pages"、"Contact"。
    *   下拉式選單："Home"、"Listings"、"Members"、"Blog"、"Pages" 具有下拉指示符號（向下箭頭圖示），表示滑鼠懸停或點擊時會出現子選單。
    *   樣式：白色文字，作用中連結樣式（若適用）。
*   **聯絡資訊：**
    *   元素：帶有圖示的文字。
    *   圖示：電話圖示。
    *   文字："+68 685 88666"。
    *   樣式：白色文字。
*   **使用者/個人資料圖示：**
    *   元素：圓形圖示（可能是圖片或字型圖示）。
    *   功能：可能連結至使用者個人資料、登入或註冊頁面/彈出視窗。
*   **「新增房產」按鈕：**
    *   元素：按鈕 (`<button>` 或 `<a>`)。
    *   文字："Add Property"。
    *   樣式：圓角，外框樣式（白色邊框、透明背景、深色頁首上的白色文字）。
    *   功能：連結至新增房產列表的頁面或彈出視窗。

### 3.2. 主視覺區塊 (Hero Section)

*   **背景：**
    *   一個家庭在房屋前的全寬圖片。
    *   圖片上可能覆蓋一層微妙的深色濾鏡以確保文字可讀性。
*   **前導標題：**
    *   元素：位於樣式化容器內的文字。
    *   文字："LET US GUIDE YOUR HOME"。
    *   樣式：小型白色文字，深色半透明圓角矩形背景。
*   **主要標題：**
    *   元素：標題 (`<h1>`)。
    *   文字："Discover a place you'll love to live"。
    *   樣式：大型、顯眼的白色文字。
*   **搜尋標籤頁：**
    *   元素：兩個標籤頁（"Sale"、"Rent"）。
    *   功能：允許使用者切換搜尋的上下文。一次只能有一個標籤頁處於作用中狀態。
    *   樣式：
        *   作用中標籤頁（圖中為 "Sale"）：白色文字，獨特的背景或底線。
        *   非作用中標籤頁（圖中為 "Rent"）：半透明白色文字。
*   **搜尋列：**
    *   容器：圓角矩形，白色背景。
    *   輸入欄位：
        *   元素：`<input type="text">`。
        *   預留位置文字："Enter Name, Keywords..."。
    *   搜尋按鈕：
        *   元素：帶有圖示的按鈕 (`<button>`)。
        *   圖示：放大鏡。
        *   樣式：圓形，米色/淺棕色背景，深色圖示。
        *   功能：根據輸入和選定的標籤頁（出售/出租）啟動搜尋。

### 3.3. 「為您推薦的房屋」區塊

*   **區塊標題：**
    *   元素：標題 (`<h2>`)。
    *   文字："Homes For You"。
    *   樣式：大型、深色文字，置中。
*   **區塊副標題：**
    *   元素：段落 (`<p>`)。
    *   文字："Based on your view history"。
    *   樣式：小型、較淺的灰色文字，置中。
*   **房產卡片輪播/滑塊：**
    *   **容器：** 容納多個房產卡片和導覽控制項。
    *   **導覽箭頭：**
        *   元素：左、右箭頭按鈕/圖示。
        *   功能：允許使用者滾動瀏覽房產卡片組。
        *   樣式：簡單的箭頭圖示，圓形或略帶圓角的背景，放置在卡片區域外部或疊加於其上。
    *   **分頁圓點：**
        *   元素：一系列小型圓點。
        *   功能：指示投影片/頁面的總數以及目前作用中的頁面。點擊圓點可導覽至對應的投影片。
        *   樣式：作用中圓點具有獨特樣式（例如：較深、較大或不同顏色的邊框）。
    *   **房產卡片 (重複組件)：**
        *   **容器：** 圓角矩形卡片，帶有細微的盒狀陰影。內容區域為白色背景。
        *   **圖片：**
            *   元素：`<img>`。
            *   樣式：頂部圓角。填滿卡片頂部的寬度。
        *   **徽章 (疊加於圖片左上角)：**
            *   "FOR SALE"：綠色背景，白色文字，圓角。
            *   "FOR RENT"：綠色背景，白色文字，圓角。
            *   "FEATURED"：米色/淺棕色背景，深色文字，圓角。
            *   *注意：* 一張卡片可以有多個徽章（例如："FOR SALE" 和 "FEATURED"）。
        *   **房產標題：**
            *   元素：標題 (例如：`<h3>` 或 `<h4>`)。
            *   範例："Skyper Pool Apartment"。
            *   樣式：深色文字。
        *   **價格：**
            *   元素：段落或 span。
            *   範例："$280,000" 或 "$250<span class="unit">/month</span>"。
            *   樣式：顯眼（例如：紅色或強調色），可能為粗體。"/month" 單位使用較小、較不顯眼的樣式。
        *   **位置：**
            *   元素：帶有圖示的段落或 span。
            *   圖示：位置圖釘圖示。
            *   範例："1020 Bloomingdale Ave"。
            *   樣式：小型、灰色文字。
        *   **房產特色 (圖示 + 文字列)：**
            *   容器：特色的 Flex 容器。
            *   項目 (每個項目包含圖示和文字)：
                *   臥室：床圖示 + 文字 (例如："4 Beds")。
                *   衛浴：浴缸圖示 + 文字 (例如："2 Baths")。
                *   面積：面積/尺規圖示 + 文字 (例如："450 sqft")。
            *   樣式：小型圖示和灰色文字。
        *   **功能：** 每張卡片應可點擊，導向至詳細的房產頁面。

## 4. 互動行為

*   **導覽下拉選單：** 滑鼠懸停或點擊時顯示子選單。
*   **搜尋標籤頁 (出售/出租)：** 點擊標籤頁會將其視覺狀態更新為作用中，並可能重新篩選或準備搜尋上下文。
*   **搜尋按鈕：** 提交搜尋查詢。
*   **輪播箭頭：** 點擊可前進或後退顯示的房產卡片組。
*   **輪播分頁圓點：** 點擊圓點可直接導航至該組卡片。
*   **房產卡片：** 點擊卡片可導航至個別房產的詳細資訊頁面。

## 5. 技術堆疊 (建議)

*   **前端框架/函式庫：** React、Vue.js 或 Angular (用於組件化架構和互動性)。
*   **樣式：**
    *   CSS 預處理器 (Sass/SCSS) 以獲得更好的組織和變數管理。
    *   或者，使用如 Tailwind CSS 的實用優先 CSS 框架。
    *   若使用 JS 框架，可採用 CSS Modules 或 Styled Components 進行作用域限定樣式。
*   **JavaScript：** ES6+ 以使用現代 JavaScript 功能。
*   **圖示：** Font Awesome、Material Icons 或自訂 SVG。
*   **輪播/滑塊函式庫：** Swiper.js、Slick Carousel 或自訂實作。
*   **建置工具：** Webpack、Vite 或 Parcel。

## 6. 資產

*   **Logo：** "JustHome" Logo 圖片/SVG。
*   **主視覺背景圖片：** 高品質的家庭與房屋圖片。
*   **房產卡片圖片：** 各種房源列表圖片。
*   **圖示：**
    *   電話
    *   使用者/個人資料
    *   下拉箭頭
    *   放大鏡
    *   輪播導覽箭頭 (左/右)
    *   位置圖釘
    *   床
    *   浴缸
    *   面積/尺規
*   **字型：** 指定用於標題和內文的網頁字型。

## 7. 資料

*   房源列表資料（用於「為您推薦的房屋」卡片）可能會從 API 擷取。每個房產物件應包含：
    *   圖片 URL
    *   標題/名稱
    *   價格 (以及單位，例如："/month" 若適用)
    *   狀態 (例如："For Sale", "For Rent")
    *   特色狀態 (布林值)
    *   地址/位置
    *   臥室數量
    *   衛浴數量
    *   面積 (sqft)
    *   詳細資訊頁面的連結/ID。
*   使用者瀏覽歷史資料（以個人化「為您推薦的房屋」）將需要後端整合和使用者追蹤。

本文件提供了基礎的技術規格。在開發過程中可能會出現更多細節。