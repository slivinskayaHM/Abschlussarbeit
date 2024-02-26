# KI-basierte Anomalieerkennung bei Sucht und Altersdepressionen über die Herzfrequenzvariabilität

Dieses Projekt umfasst einen Autoencoder, der mithilfe von HRV-Metriken gesunder Personen trainiert wird. Des Weiteren sind Programme enthalten, die RR-Intervalle in Millisekunden umwandeln und die HRV-Metriken daraus ermitteln.

## Beschreibung

Das Projekt besteht aus drei ausführbaren Dateien und Dateien die Daten enthalten, welche im Folgenden näher betrachtet werden.

### Ausführbare Dateien

Die erste ausführbare Datei ist **RRIntervalleInSekundenZuMs.py**. Hier werden RR-Intervalle aus Ordnern extrahiert, von Sekunden in Millisekunden umgewandelt und in .txt-Datein abgespeichert. Es handelt sich hierbei um einen Schritt die Daten vor der Nutzung zu vereinheitlichen. Diese Datei wurde verwendet, um die Daten aus den Ordnern **/Daten/MMASH** und **/Daten/HRV_bipolarund_schizophren** zu vereinheitlichen.

Die nächste ist **RRIntervallToHRV.py**. Aus den erstellten RR-Intervallen in Millisekunden werden hier HRV-Metriken berechnet und in eine separate Datei gesammelt abgespeichert. Dieser Vorgang wurde genutzt, um das Trainingsset und das Vergleichsdatenset zu erstellen. Für das Trainingsdatenset wurden hierfür alle Dateien aus den Ordnern **/Daten/MMASH Data Umgewandelt** und **/Daten/PhysioNet** verarbeitet und gesammelt als **healthy_hrv_metrics.csv** abgespeichert. Für das Vergleichsdatenset wurden hier die Daten aus dem Ordner **/Daten/HRV_BS** verwendet und in der Datei **/TestSetsWithoutLabels/BS_hrv_metrics.csv** gesichert.

Schließlich ist die Datei **Autoencoder.py** ebenfalls ausführbar. In dieser Datei befindet sich die Künstliche Intelligenz, welche nach dem Training die Anomalieerkennung anhand der Testdaten durchführt.

### Dateien mit Daten

Der Ordner **/Daten** enthält Rohdaten mit RR-Intervallen.

In **/TestSetsWithoutLabels** sind alle HRV-Metriken der jeweiligen Diagnosen separat zu finden. Dabei Enthält die Datei **/TestSetsWithoutLabels/BS_hrv_metrics.csv** die HRV-Metriken bipolarer und schizophrener Personen. **/TestSetsWithoutLabels/testSetAdd.csv** enthält die HRV-Metriken abhängiger Personen, während **/TestSetsWithoutLabels/testSetDepr.csv** die Werte Altersdepressiver Menschen enthält.

Die Hauptordnerstruktur enthält die Dateien **healthy_hrv_metrics.csv**, **bipolarSchizophrenicTestSet.csv** und **testSet.csv**. **healthy_hrv_metrics.csv** ist hierbei die Datei mit den Trainingsdaten. Die beiden anderen Dateien sind Testsets, welche sich aus Anomalien und Daten gesunder Personen zusammensetzen. Des Weiteren sind die Dateien der Testsets gelabelt, wobei 0 als normal gilt, während 1 eine Anomalie darstellt. Die Anomalien der Datei **testSet.csv** setzen sich aus den beiden Dateien **/TestSetsWithoutLabels/testSetAdd.csv** und **/TestSetsWithoutLabels/testSetDepr.csv** zusammen. Die Anomalien der Datei **bipolarSchizophrenicTestSet.csv** sind die Werte aus der Datei **/TestSetsWithoutLabels/BS_hrv_metrics.csv**.

## Installation

Dieses Projekt läuft mit der **python version 3.11.7**. Die installierten Bibliotheken sind in der Datei **requirements.txt** zu finden. 

## Notiz 

Die virtuelle Umgebung, sowie die Daten der Ordner **/Daten/MMASH** und **/Daten/HRV_bipolarund_schizophren** wurden entfernt, da die Verschicjung per Email sonst nicht möglich war. Die Daten dieser Ordner waren Rohdaten, die später in passende Schemata umgewnandelt wurden, wie oben beschrieben. Diese sind zitiert in der Bachelorarbeit. Ansonsten können Sie sich gerne bei mir melden, dann lasse ich Sie Ihnen zu kommen.