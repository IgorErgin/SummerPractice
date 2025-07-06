# Отчет по исследованию архитектур нейронных сетей

## Введение

Цель данного проекта — исследование и сравнение различных архитектур нейронных сетей на задачах классификации изображений с использованием датасетов MNIST и CIFAR-10. В ходе работы были реализованы и проанализированы полносвязные сети, сверточные нейронные сети (CNN), а также кастомные слои и Residual блоки. Основное внимание уделялось влиянию архитектурных решений на производительность, время обучения и стабильность моделей.

---

## Задание 1: Сравнение CNN и полносвязных сетей

### Выводы
- **MNIST**:
  - Полносвязная сеть показала приемлемую точность, но уступила CNN из-за отсутствия способности эффективно обрабатывать пространственные данные.
  - Простая CNN обеспечила более высокую точность благодаря локальным связям и меньшему количеству параметров.
  - CNN с Residual блоками еще больше улучшила результаты, позволив строить глубокие сети без деградации производительности.
   ![image](https://github.com/user-attachments/assets/0c98488f-afbc-4848-8c87-4af5f76fb21b)
   ![image](https://github.com/user-attachments/assets/d20835b5-9145-4db9-8c8f-3e9ef6ae9902)
   ![image](https://github.com/user-attachments/assets/af7f3bd6-1605-482a-b9d8-10258a3984f3)
   ![image](https://github.com/user-attachments/assets/f2db94ee-3b4b-4b2f-b528-dd651217dd05)
   ![image](https://github.com/user-attachments/assets/9dbf2a6c-b272-43c2-804a-c005f0f9f9f2)
   ![image](https://github.com/user-attachments/assets/7bf3baa5-9199-4885-b6d9-eef2e719838a)
   ![image](https://github.com/user-attachments/assets/cb2ef5db-d69d-4500-bcd7-edfc4d13c4e3)
   ![image](https://github.com/user-attachments/assets/f2edf122-454b-4213-a9dd-5ce9d849a29b)


- **CIFAR-10**:
  - Полносвязная сеть показала низкую точность и склонность к переобучению из-за высокой сложности изображений.
  - Простая CNN справилась лучше, но недостаточно для сложных данных.
  - CNN с Residual блоками значительно повысила точность и обобщающую способность за счет глубоких архитектур и регуляризации.
 
    ![image](https://github.com/user-attachments/assets/7b9b60f3-f5cd-48f7-a2a7-9f8db251c27a)
    ![image](https://github.com/user-attachments/assets/1cd1bf9c-cc03-4c92-a1ca-33b5df513415)
    ![image](https://github.com/user-attachments/assets/db48065e-e78e-46d7-8dd6-4b6d2712e673)
    ![image](https://github.com/user-attachments/assets/3e7df051-d54a-4a11-b82a-8c5a602fee4c)
    ![image](https://github.com/user-attachments/assets/24fdeea8-8a1d-417b-ae9f-aad5edbdf4df)
    ![image](https://github.com/user-attachments/assets/f0c9b3eb-9181-4333-9d5f-fadd755e43fa)
    ![image](https://github.com/user-attachments/assets/7a18b543-40d8-4d3e-87fa-4d0056c04fa7)
    ![image](https://github.com/user-attachments/assets/06c1fd08-ffcb-46ec-bceb-747150117253)
    ![image](https://github.com/user-attachments/assets/8c05a167-eb05-4473-8911-fec5348f9b0e)
    ![image](https://github.com/user-attachments/assets/a54acfe5-9d7b-436f-9951-602395fa358b)
    ![image](https://github.com/user-attachments/assets/8801adcf-471d-4ad8-9a30-8f388c7b5107)



**Общий вывод**: CNN превосходят полносвязные сети в задачах классификации изображений благодаря инвариантности к сдвигам и локальной обработке данных. Residual блоки делают глубокие CNN более эффективными и устойчивыми.

---

## Задание 2: Анализ архитектур CNN

### Выводы
- **Влияние размера ядра свертки**:
  - Ядра 3x3 оказались наиболее сбалансированными по точности и вычислительной нагрузке.
  - Большие ядра (5x5, 7x7) увеличили время обучения и иногда ухудшали результаты из-за избыточной детализации.
  - Комбинация 1x1 + 3x3 сверток сократила количество параметров, сохранив качество предсказаний.

![download](https://github.com/user-attachments/assets/2d715c67-b013-432c-8695-3d5fa6c620d4)
    
![download](https://github.com/user-attachments/assets/1f679d01-256c-4e85-94ee-ee259b45547e)
![download](https://github.com/user-attachments/assets/00217b45-1a28-4113-948a-179c857d16a4)
![download](https://github.com/user-attachments/assets/ffdd2b74-9cb5-4402-95b1-440564d314ab)
![download](https://github.com/user-attachments/assets/7bec2eb2-8b6c-43a2-bd35-80e379e252d3)
![download](https://github.com/user-attachments/assets/26110e69-734e-465c-be47-a250fc3bf320)
![download](https://github.com/user-attachments/assets/46bb50b8-2699-417f-9e41-321ad2d8099b)
![download](https://github.com/user-attachments/assets/6dea3023-0f29-4de7-bf87-e9cc4c0446a6)
![download](https://github.com/user-attachments/assets/d9015512-c4af-4e4b-8fb9-a353183e7c87)
![download](https://github.com/user-attachments/assets/88bde021-ffeb-4fff-a329-062d9523e113)
![download](https://github.com/user-attachments/assets/19244fb2-cd44-4456-8a88-5e9fe0809868)



- **Влияние глубины CNN**:
  - Неглубокие сети (2 слоя) были быстрыми, но недостаточно точными.
  - Глубокие сети (6+ слоев) показали лучшую производительность, но без Residual связей страдали от исчезающего градиента.
  - Residual связи позволили эффективно обучать глубокие архитектуры, улучшив точность.
 
![download](https://github.com/user-attachments/assets/0d63f75d-7bce-4541-8644-6e8bb9dd73ea)
![download](https://github.com/user-attachments/assets/5b294220-17fe-4f4d-99c6-64a0aedc08b6)
![download](https://github.com/user-attachments/assets/dd46204e-38fb-478c-970d-be7582850efe)
![download](https://github.com/user-attachments/assets/a2693c5a-aed6-4688-b1fd-c8ac4de29631)
![download](https://github.com/user-attachments/assets/17b64667-5c1e-4049-b7bb-f80dff74ef9c)
![download](https://github.com/user-attachments/assets/0a07ce6b-dcc4-4dad-87cb-50b65e017bee)
![download](https://github.com/user-attachments/assets/cafab099-c358-4d0c-b25d-271fce848c41)
![download](https://github.com/user-attachments/assets/add5bfa8-c1b7-4888-8258-8df83138aafc)
![download](https://github.com/user-attachments/assets/01e1faf7-b36f-4114-b181-60e4e41a0469)
![download](https://github.com/user-attachments/assets/96816bc0-2140-4cc7-ab4a-201138e605b8)
![download](https://github.com/user-attachments/assets/c9861f82-ea14-41b5-b67d-5f9f40e9af1f)
![download](https://github.com/user-attachments/assets/13ef9a8b-1107-4b7a-8b0d-49d00a1d2035)
![download](https://github.com/user-attachments/assets/1f0f0c88-a50f-459f-99f8-0e633224df5c)
![download](https://github.com/user-attachments/assets/dca2ddf6-7521-4b37-93e0-5ad367af123d)
![download](https://github.com/user-attachments/assets/6fcd399f-a928-45fa-8236-c4ba6b430d9f)


**Общий вывод**: Ядра 3x3 и глубокие сети с Residual связями обеспечивают оптимальную производительность для классификации изображений.

---

## Задание 3: Кастомные слои и эксперименты

### Выводы
- **Кастомные слои**:
  - Реализация собственных слоев (свертка, attention, pooling) дала гибкость в настройке моделей под задачу.
  - Однако такие слои требуют точной реализации backward проходов, что усложняет разработку.
![download](https://github.com/user-attachments/assets/50c9dea9-4075-471a-96a2-beba4bbbc638)
![download](https://github.com/user-attachments/assets/9e5cae0b-de31-4764-a0e4-69d2b699da21)
![download](https://github.com/user-attachments/assets/e696cd70-f719-442a-a586-da796c7144c8)


    
- **Эксперименты с Residual блоками**:
  - Базовый Residual блок улучшил обучение глубоких сетей, но увеличил число параметров.
  - Bottleneck Residual блок снизил вычислительную нагрузку, сохранив точность.
  - Wide Residual блок повысил стабильность за счет увеличения ширины слоев.
 
![download](https://github.com/user-attachments/assets/db1e1bb9-7074-4826-be82-2dde9f2b8907)
![download](https://github.com/user-attachments/assets/ca91ac9b-6cd9-4e8f-afc6-b15ecf27eed5)
![download](https://github.com/user-attachments/assets/b025e37e-7ab1-4b08-90ad-7400c0ca7b55)
![download](https://github.com/user-attachments/assets/2e396727-03c4-4c6d-9640-54f661ac5966)
![download](https://github.com/user-attachments/assets/dc3cbc1e-6f73-472c-823b-7f7e415415eb)
![download](https://github.com/user-attachments/assets/9abfac9e-d791-41a6-ba39-8743fab0e8bf)
![download](https://github.com/user-attachments/assets/ab9c872d-45d8-4ecf-a371-a2b7218aaa31)
![download](https://github.com/user-attachments/assets/c4af27f1-0239-4e53-b7b0-12f2748b654c)

**Общий вывод**: Кастомные слои полезны для специфических задач, но требуют аккуратной реализации. Bottleneck Residual блоки — оптимальный выбор для глубоких сетей с ограниченными ресурсами.

---

## Заключение

Исследование подтвердило, что CNN с Residual блоками — наиболее эффективная архитектура для классификации изображений, особенно на сложных датасетах. Кастомные слои добавляют гибкости, но требуют тщательной реализации. Оптимальная архитектура зависит от задачи и доступных ресурсов.
