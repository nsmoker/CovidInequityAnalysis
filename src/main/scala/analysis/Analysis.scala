package analysis

import org.apache.spark.ml.classification.{DecisionTreeClassifier, FMClassifier, GBTClassifier, LinearSVC, LogisticRegression, LogisticRegressionModel, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassifier}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.types.{DoubleType, StringType}
import swiftvis2.plotting.renderer.SwingRenderer
import swiftvis2.plotting.styles.ScatterStyle
import swiftvis2.plotting.{BlackARGB, BlueARGB, ColorGradient, GreenARGB, MagentaARGB, NumericAxis, Plot, PlotIntSeries, RedARGB, YellowARGB}
import swiftvis2.spark.doubles

object Analysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("Outcomes Analysis").getOrCreate()
    import spark.implicits._

    spark.sparkContext.setLogLevel("WARN")

    // Loading in and preparing data

    val stateThing = spark.read.option("header", true).option("inferschema", true).csv("data/state_names_toabbrs.csv")
      .withColumnRenamed("State", "sname")

    val geoData = spark.read.option("header", true).option("inferschema", true).csv("data/Geocodes_USA_with_Counties.csv")
      .withColumn("join_col", concat($"county", when($"state" === "LA", " ").otherwise(" County"), lit(", "), $"state"))

    val covidData = spark.read.option("header", true).option("inferschema", true).csv("data/covid-us-counties.csv")

    val otherOutcomes = spark.read.option("header", true).option("inferschema", true).csv("data/outcome_better.csv").drop("county")

    val blsData = spark.read.option("inferschema", true).option("header", true).option("delimiter", "\t").csv("data/la/la.data.64.County")

    val blsSeriesData = spark.read.option("inferschema", true).option("header", true).option("delimiter", "\t").csv("data/la/la.series")

    val blsAreaData = spark.read.option("inferschema", true).option("header", true).option("delimiter", "\t").csv("data/la/la.area")

    val blsCountiesData = blsAreaData.filter($"area_type_code" === "F").join(blsSeriesData, "area_code")

    val blsJoined = blsData.join(blsCountiesData, "series_id").filter($"year" === 2020 && $"measure_code" === 3)

    val blsWithGeo = blsJoined.join(geoData, $"join_col" === $"area_text")

    val joinedFresh = otherOutcomes
//      .join(covidData, $"County FIPS Code" === $"fips").filter($"date".startsWith("2020-10")).dropDuplicates("fips")
      .join(stateThing, $"state" === $"sname")
      .withColumn("join_col", concat($"County Name", when($"sname" === "Louisiana", " Parish, ").otherwise(" County, "), $"Code"))

    val allJoined = joinedFresh.join(blsWithGeo, "join_col").withColumn("value", $"value".cast(DoubleType)).cache()

    val marchData = allJoined.filter($"period" === "M03").cache()

    val aprilData = allJoined.filter($"period" === "M04").cache()

    val numericCols = allJoined.schema.fields.filterNot(_.dataType == StringType).map(_.name)

    val corVA = new VectorAssembler().setInputCols(numericCols).setOutputCol("corVect").setHandleInvalid("skip")

    val withCorVectMarch = corVA.transform(marchData).cache()

    val withCorVectApril = corVA.transform(aprilData).cache()

    val Row(corMarch: Matrix) = Correlation.corr(withCorVectMarch, "corVect").head()

    val Row(corApril: Matrix) = Correlation.corr(withCorVectApril, "corVect").head()

    val corArrayMarch = corMarch.colIter.toArray.zip(numericCols)

    val corArrayApril = corApril.colIter.toArray.zip(numericCols)

    val unempCorrMarch = corArrayMarch.find(_._2 == "value").get._1.toArray.zip(numericCols).sorted

    val unempCorrApril = corArrayApril.find(_._2 == "value").get._1.toArray.zip(numericCols).sorted

    val diffData = marchData.drop("period", "notes", "footnote_codes").withColumnRenamed("value", "unemp_march").join(
      aprilData.drop("period", "notes", "footnote_codes").withColumnRenamed("value", "unemp_april"), aprilData.columns.filter(s => !Seq("value", "period", "notes", "footnote_codes").contains(s))
    ).withColumn("diff_col", $"unemp_april" - $"unemp_march")
      .cache()

    println(diffData.count())

    diffData.show(false)

    marchData.unpersist()

    aprilData.unpersist()

    val diffNums = diffData.schema.fields.filterNot(_.dataType == StringType).map(_.name)

    corVA.setInputCols(diffNums)

    val withCorVectDiff = corVA.transform(diffData).cache()

    val Row(corDiff: Matrix) = Correlation.corr(withCorVectDiff, "corVect").head()

    val corArrayDiff = corDiff.colIter.toArray.zip(diffNums)

    val unempCorrDiff = corArrayDiff.find(_._2 == "diff_col").get._1.toArray.zip(diffNums).sorted

    println("Diff: " + unempCorrDiff.mkString("Array(", ", \n", ")"))

    val classy = new GBTClassifier().setFeaturesCol("classVect").setLabelCol("diff_class").setPredictionCol("pred_diff_class").setWeightCol("weights")

    val classyVA = new VectorAssembler().setInputCols(Array("Absolute Upward Mobility", "Interquartile Income Range", "Child Income P25", "Median Parent Income")).setOutputCol("classVect").setHandleInvalid("skip")

    val withClassyVect = classyVA.transform(diffData)

    val withClass = withClassyVect.withColumn("diff_class",
        when($"diff_col" <= 10, 0)
        .otherwise(1))
      .withColumn("weights", when($"diff_class" === 1, 1.0).otherwise(0.75))

    val classyModel = classy.fit(withClass)

    val withPred = classyModel.transform(withClass).cache()

    val classyCols = ColorGradient(0.0 -> BlueARGB, 1.0 -> RedARGB)

    val actuals = classyCols(doubles(withPred, $"diff_class"))

    val preds = classyCols(doubles(withPred, $"pred_diff_class"))

    val classyPlotX = doubles(withPred, $"longitude")
    val classyPlotY = doubles(withPred, $"latitude")

    val classyStyle = (cols: PlotIntSeries) => ScatterStyle(classyPlotX, classyPlotY).coloredBy(cols, Seq("diff <= 10" -> BlueARGB, "diff > 10" -> RedARGB))

    val classyPlot = Plot.row(Seq(classyStyle(actuals), classyStyle(preds)), "Actual vs. predicted classes for U.S. county unemployment rate differences after COVID", "Longitude", "Latitude")
      .withModifiedAxis("nx", "nx", (ax: NumericAxis) => ax.updatedMin(-130))
      .withModifiedAxis("y", "y", (ax: NumericAxis) => ax.updatedMin(25))
      .withGeneratedLegend()

    SwingRenderer(classyPlot, 1920, 1080, true)

    println("Prediction success: " + withPred.filter($"diff_class" === $"pred_diff_class").count().toDouble / withPred.count().toDouble)

    println(classyModel.featureImportances.toArray.mkString("Array(", ", ", ")"))

    println()

//    val cg = ColorGradient(35.0 -> RedARGB, 42.5 -> MagentaARGB, 50.0 -> BlueARGB)
//
//    val geoMobilCol = doubles(diffData, $"Absolute Upward Mobility")
//
//    val geoMobilX = doubles(diffData, $"longitude")
//
//    val geoMobilY = doubles(diffData, $"latitude")
//
//    val style = ScatterStyle(geoMobilX, geoMobilY).coloredByGradient(cg, geoMobilCol, "Absolute upward mobility", false)
//
//    SwingRenderer(Plot.simple(style, "Absolute Upward Mobility, Geographical", "Longitude", "Latitude").withGeneratedLegend()
//      .withModifiedAxis("x", "x", (ax: NumericAxis) => ax.updatedMin(-135.0))
//      .withModifiedAxis("y", "y", (ax: NumericAxis) => ax.updatedMin(22.0)), 1920, 1080, true)

//    val numericCols = joinedFresh.schema.fields.filterNot(_.dataType == StringType).map(_.name)
//
//    val corVA = new VectorAssembler().setInputCols(numericCols).setOutputCol("corVect").setHandleInvalid("skip")
//
//    val withCorVect = corVA.transform(joinedFresh).cache()
//
//    val Row(cor: Matrix) = Correlation.corr(withCorVect, "corVect").head()
//
//    val corArray = cor.colIter.toArray
//
//    val deathsCor = corArray.last.toArray.zip(numericCols)
//
//    val casesCor = corArray(corArray.length - 1).toArray.zip(numericCols)
//
//    val topDeathCors = deathsCor.sortBy(_._1)
//    val topCasesCors = casesCor.sortBy(_._1)
//
//    val clusterVA = new VectorAssembler().setInputCols(Array("Interquartile Income Range", "Share Between p25 and p75")).setOutputCol("predVect").setHandleInvalid("skip")
//
//    val withClusterVect = clusterVA.transform(joinedFresh)
//
//    val kmeans = new KMeans().setFeaturesCol("predVect").setPredictionCol("predCases").setK(2).fit(withClusterVect.filter($"cases" <= 20000))
//
//    val withClusterPred = kmeans.transform(withClusterVect.filter($"cases" <= 20000))
//
//    val lr = new LinearRegression().setFeaturesCol("predVect").setPredictionCol("predCases").setLabelCol("cases")
//
//    val fitted = lr.fit(withClusterVect.filter($"cases" <= 20000))
//
//    val cg = ColorGradient(0.0 -> RedARGB, 1.0 -> BlueARGB, 2.0 -> GreenARGB, 3.0 -> YellowARGB, 4.0 -> BlackARGB)
//
//    val clusterData = doubles(withClusterPred, $"cases")
//
//    val colors = cg(doubles(withClusterPred, $"predCases"))
//
//    val style = ScatterStyle(clusterData, clusterData).coloredBy(colors, Seq("Predicted under 3k" -> RedARGB, "Predicted more than 3k" -> BlueARGB))
//
//    val countAbove3K = withClusterPred.filter($"cases" >= 3000).count().toDouble
//    val countBelow3K = withClusterPred.filter($"cases" < 3000).count().toDouble
//
//    val countRightClusterZero = withClusterPred.filter($"cases" < 3000 && $"predCases" === 0).count().toDouble
//    val countRightClusterOne = withClusterPred.filter($"cases" >= 3000 && $"predCases" === 1).count().toDouble
//
//    val fracRightClusterZero = countRightClusterZero / countBelow3K
//    val fracRightClusterOne = countRightClusterOne / countAbove3K
//
//    val fracRightOverall = (countRightClusterOne + countRightClusterZero) / (withClusterPred.count().toDouble)
//
//    println(fracRightClusterZero)
//    println(fracRightClusterOne)
//    println(fracRightOverall)
//
//
//    SwingRenderer(Plot.simple(style, "KMeans clustering: coronavirus cases clustered by interquartile income ranges and income shares", "Cases")
//      .withModifiedAxis("x", "x", (ax: NumericAxis) => ax.updatedMax(20000))
//        .withModifiedAxis("y", "y", (ax: NumericAxis) => ax.updatedMax(20000)).withGeneratedLegend()
//        , makeMain = true)

    //    sprawlData.show(false)

//    val belMed = allOutcomeData.select($"kid_pooled_pooled_blw_p50_n", $"county", $"state").withColumn("fips", concat($"county", $"state"))
//
//    val joined = belMed.join(sprawlData, $"fips" === $"CountyFips").withColumn("compositeindex2010", $"compositeindex2010".cast(DoubleType))

//    val withOnlyLatestCases = joined.filter($"date" === "2020-10-20")

//    println(joined.count())
//
//    val numericCols = joined.schema.fields.filterNot(_.dataType == StringType).map(_.name).filter(s => s != "county" && s != "fips" && s != "state"
//      && !s.endsWith("count"))
//
//    val selectedCols = Array("kid_pooled_pooled_blw_p50_n", "densityfactor", "compositeindex2010")
//
//    val corVA = new VectorAssembler().setInputCols(selectedCols).setOutputCol("corVect").setHandleInvalid("skip")
//
//    val withCorVect = corVA.transform(joined).cache()
//
//    val Row(cor: Matrix) = Correlation.corr(withCorVect, "corVect").head()
//
//    val corArray = cor.colIter.toArray
//
//    val deathsCor = corArray.head.toArray.zip(selectedCols)
//
//    val casesCor = corArray(1).toArray.zip(selectedCols)
//
//    val topDeathCors = deathsCor.sortBy(_._1).takeRight(5)
//    val topCasesCors = casesCor.sortBy(_._1).takeRight(5)
//
//    println(topDeathCors.mkString("Array(", ", ", ")"))
//    println(topCasesCors.mkString("Array(", ", ", ")"))

    spark.sparkContext.stop()
  }
}
