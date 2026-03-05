document.addEventListener('DOMContentLoaded', () => {
    const userRequirement = document.getElementById('userRequirement');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const resultsScreen = document.getElementById('resultsScreen');
    const processingStatus = document.getElementById('processingStatus');
    const recommendationList = document.getElementById('recommendationList');
    const mainContent = document.getElementById('mainContent');
    const chips = document.querySelectorAll('.chip');

    // 初始化 textarea 高度
    userRequirement.style.height = "auto";
    userRequirement.style.height = (userRequirement.scrollHeight) + "px";

    // 處理輸入框自動長高
    userRequirement.addEventListener('input', function () {
        this.style.height = 'auto'; // 重置高度
        this.style.height = (this.scrollHeight) + 'px'; // 設為內容高度

        // 限縮最高高度
        if (this.scrollHeight > 120) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
            // 自動滾到底部確保輸入框在視野內
            mainContent.scrollTop = mainContent.scrollHeight;
        }
    });

    let debounceTimer;

    // 監聽文字輸入事件 (即時響應)
    userRequirement.addEventListener('input', () => {
        const text = userRequirement.value.trim();

        // 如果清空內容，則顯示歡迎畫面，隱藏結果
        if (!text) {
            resultsScreen.style.display = 'none';
            welcomeScreen.style.display = 'flex';
            return;
        }

        // 當有字輸入時，切換為結果畫面
        welcomeScreen.style.display = 'none';
        resultsScreen.style.display = 'block';

        // 每次打字時，顯示小型的運算動畫
        processingStatus.style.display = 'flex';
        // 降低透明度模擬正在更新
        recommendationList.style.opacity = '0.4';

        // 保證結果區是在畫面上方的，讓使用者不必下滑
        mainContent.scrollTop = 0;

        // 清除上一次的計時器
        clearTimeout(debounceTimer);

        // 設定 600ms 後使用者如果沒繼續打字，才執行尋找
        debounceTimer = setTimeout(() => {
            // 隱藏運算動畫
            processingStatus.style.display = 'none';

            // 模擬更新結果
            updateMockResults(text);

            // 恢復透明度
            recommendationList.style.opacity = '1';

        }, 600);
    });

    // 點擊建議標籤快速輸入
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            userRequirement.value = chip.textContent;
            // 觸發 input 事件與自動拉高
            userRequirement.dispatchEvent(new Event('input'));
            userRequirement.focus(); // 輸入框保持 Focus
        });
    });

    // 動態變更推薦結果
    function updateMockResults(inputText) {
        // 從現有 DOM 抓取卡片來模擬打亂順序
        const cards = Array.from(recommendationList.querySelectorAll('.property-card'));

        for (let i = cards.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [cards[i], cards[j]] = [cards[j], cards[i]];
        }

        recommendationList.innerHTML = '';
        cards.forEach((card, index) => {
            // Reset state
            card.classList.remove('top-match');
            const badge = card.querySelector('.badge');

            if (index === 0) {
                card.classList.add('top-match');
                badge.classList.add('premium');
                badge.textContent = `配對度 ${Math.floor(95 + Math.random() * 5)}%`;

                // AI 推薦理由已移除
            } else {
                badge.classList.remove('premium');
                badge.textContent = `配對度 ${Math.floor(80 + Math.random() * 14)}%`;
            }

            recommendationList.appendChild(card);
        });
    }

    // 發送按鈕的點擊與鍵盤 Enter 直接觸發防抖立刻執行
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', () => {
            clearTimeout(debounceTimer);
            const text = userRequirement.value.trim();
            if (text) updateMockResults(text);
            processingStatus.style.display = 'none';
            recommendationList.style.opacity = '1';
        });
    }
});
