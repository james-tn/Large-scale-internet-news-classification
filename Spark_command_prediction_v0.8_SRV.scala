
cd C:\spark\spark-1.6.1-bin-hadoop2.6\bin
spark-shell --jars stanford-corenlp-3.6.0-models.jar,stanford-corenlp-3.6.0.jar,slf4j-api,slf4j-simple.jar,joda-time.jar,jollyday.jar,ejml-0.23.jar

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import java.util.Properties
import scala.collection.JavaConverters._
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SaveMode}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
var lock : AnyRef = new Object()  with Serializable
def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)

  }


def isOnlyLetters(str: String): Boolean = {
    // While loop for high performance
    var i = 0
    while (i < str.length) {
      if (!Character.isLetter(str.charAt(i))) {
        return false
      }
      i += 1
    }
    true
  }

def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
    : Seq[String] = {
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }

 def documentFrequenciesDistributed(docTermFreqs: RDD[HashMap[String, Int]], numTerms: Int)
      : Array[(String, Int)] = {
    val docFreqs = docTermFreqs.flatMap(_.keySet).map((_, 1)).reduceByKey(_ + _, 15)
    val ordering = Ordering.by[(String, Int), Int](_._2)
    docFreqs.top(numTerms)(ordering)
  }




def trainModel(sqlContext: SQLContext, url: String, prop: java.util.Properties, stopWords:Set[String], headerWeight: Int, kwWeight: Int): (NaiveBayesModel, Map[String, Int], Map[String, Double], Map[String, Int]) = {



val news= sqlContext.read.jdbc(url,"news",prop)
val category= news.select("Category").rdd.map(_.toString)
val newsDocs = news.select("Category", "Header", "KW", "Abstract", "Content").map(t => {

if (t(3)!=null) 
(Option(t(0)).getOrElse(" ").toString, Option(t(1)).getOrElse("").toString*headerWeight +" " + Option(t(2)).getOrElse("").toString*kwWeight +" " +Option(t(3)).getOrElse("").toString)
else 
(Option(t(0)).getOrElse(" ").toString, Option(t(1)).getOrElse("").toString*headerWeight +" " + Option(t(2)).getOrElse("").toString*kwWeight +" " +Option(t(4)).getOrElse("").toString)
})
println("newsDocs first:" +newsDocs.first)




val lemmatized = newsDocs.mapPartitions(iter => {
      val pipeline = createNLPPipeline()
      iter.map{ case(cat, contents) => (cat, plainTextToLemmas(contents, stopWords, pipeline))}
    })
	

val docTermFreqs = lemmatized.mapValues(terms => {
      val termFreqsInDoc = terms.foldLeft(new HashMap[String, Int]()) {
        (map, term) => map += term -> (map.getOrElse(term, 0) + 1)
      }
      termFreqsInDoc
    })

    docTermFreqs.cache()

    
  
    val docFreqs = documentFrequenciesDistributed(docTermFreqs.map(_._2), 8000)
    println("Number of terms: " + docFreqs.size)
 val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap()

 val numDocs = docIds.size
   def inverseDocumentFrequencies(docFreqs: Array[(String, Int)], numDocs: Int)
    : Map[String, Double] = {
    docFreqs.map{ case (term, count) => (term, math.log(numDocs.toDouble / count))}.toMap
  }

 
   val idfs = inverseDocumentFrequencies(docFreqs, numDocs)

    // Maps terms to their indices in the vector
    val idTerms = idfs.keys.zipWithIndex.toMap
 

    val bIdfs = sc.broadcast(idfs).value
    val bIdTerms = sc.broadcast(idTerms).value

    val termDocMatrix= docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bIdTerms.contains(term)
      }.map{
        case (term, freq) => (bIdTerms(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      Vectors.sparse(bIdTerms.size, termScores)
    })

   
val cat= docTermFreqs.map(_._1)

val labels= cat.distinct.collect().zipWithIndex.toMap

println("before zip")

val newsData = cat zip termDocMatrix

val data = newsData.map { case (cat, vector) => LabeledPoint (labels(cat), vector)}


val model = NaiveBayes.train(data, lambda = 0.1)

(model, labels, bIdfs,bIdTerms)

}

def predict(labels: Map[String, Int], model: NaiveBayesModel, sqlContext: SQLContext, url: String, prop: java.util.Properties, stopWords:Set[String], lastRecord: Int, headerWeight: Int, kwWeight: Int, bIdfs: Map[String, Double], bIdTerms: Map[String, Int]): Double ={
this.synchronized {

val newsTable= sqlContext.read.jdbc(url,"newsInput",prop)
newsTable.registerTempTable("newsInput")
val sqlString = "Select ID, Date, Header, Abstract, Content, Company, Source from newsInput where ID > '" + lastRecord +"'"
val newsRecords= sqlContext.sql(sqlString)
val newsRecordRDD= newsRecords.map(t => (Option(t(0)).getOrElse("NA").toString, Option(t(1)).getOrElse("NA").toString, Option(t(2)).getOrElse("NA").toString, Option(t(3)).getOrElse("NA").toString, Option(t(4)).getOrElse("NA").toString, Option(t(5)).getOrElse("NA").toString, Option(t(6)).getOrElse("NA").toString))


if (newsRecords.rdd.isEmpty) 
{ println("no new data")
return lastRecord
}
newsRecordRDD.cache() 

val newsDocs = newsRecordRDD.map(t => 
{
if (t._4!="NA") 
(t._1, t._3*headerWeight +" " + t._4)
else
(t._1, t._3*headerWeight +" " + t._5)
})

val lemmatized = newsDocs.mapPartitions(iter => {
      val pipeline = createNLPPipeline()
      iter.map{ case(cat, contents) => (cat, plainTextToLemmas(contents, stopWords, pipeline))}
    })
	


val docTermFreqs = lemmatized.mapValues(terms => {
      val termFreqsInDoc = terms.foldLeft(new HashMap[String, Int]()) {
        (map, term) => map += term -> (map.getOrElse(term, 0) + 1)
      }
      termFreqsInDoc
    })


     val termDocMatrix= docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bIdTerms.contains(term)
      }.map{
        case (term, freq) => (bIdTerms(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      Vectors.sparse(bIdTerms.size, termScores) 	
    })




val labelNums = termDocMatrix.map(p => Option(model.predict(p)).getOrElse("0").toString)

val k = labels.map(_.swap)
val labelText = labelNums.map(p => k(p.toDouble.toInt))

println("new record count 2: "+ newsRecords.count)
println("Labeltext count:"+ labelText.count)
println("NewsRecord RDD count 2:"+ newsRecordRDD.count)

val result = labelText zip newsRecordRDD
println("before result first ")

val output = result.map{case (x,y) => (x, y._1, y._2, y._3, y._4, y._5, y._6, y._7)}
import sqlContext.implicits._	
val resultDF = output.toDF("Category", "ID", "Date", "Header", "Abstract", "Content", "Company", "Source")
println("before write ")
resultDF.write.mode(SaveMode.Append).jdbc(url, "newsOutput", prop)
println("write ok")
val newLastRecordtemp = newsRecordRDD.takeOrdered(1)( Ordering[Double].reverse.on( x => x._1.toDouble ))
val newLastRecord = newLastRecordtemp(0)._1.toDouble

println("last record in inputNews is "+newLastRecord )

newLastRecord 

}

}


val stopWords = sc.broadcast(scala.io.Source.fromFile("stopwords.txt").getLines().toSet)value
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val url="jdbc:sqlserver://50.23.132.180"
val prop = new java.util.Properties

prop.setProperty("user","sa")
	
prop.setProperty("password","Abcd@123")
prop.setProperty("databaseName","AON")
var lastRecord =0


val headerWeight = 5
val kwWeight = 10

def getLastModelRecord() : Double= {
val trainTable= sqlContext.read.jdbc(url,"news",prop)
trainTable.registerTempTable("trainTable")
Option(sqlContext.sql("Select Max(ID) from trainTable").first()(0)).getOrElse("0").toString.toDouble

}

var lastModelRecord = getLastModelRecord()
var newLastModelRecord = lastModelRecord 
var modelAndCat = trainModel(sqlContext, url, prop, stopWords, headerWeight, kwWeight)
var model = modelAndCat._1
var labels = modelAndCat._2
var bIdfs = modelAndCat._3
var bIdTerms =modelAndCat._4



while(true) {


if (newLastModelRecord != lastModelRecord) {
println(" need to retrain ")
modelAndCat = trainModel(sqlContext, url, prop, stopWords, headerWeight, kwWeight)
model = modelAndCat._1
labels = modelAndCat._2
bIdfs = modelAndCat._3
bIdTerms =modelAndCat._4
lastModelRecord= newLastModelRecord
}
else
println("no need to retrain: max in news table is "+ newLastModelRecord)

val newLastRecord = predict(labels, model, sqlContext, url, prop, stopWords, lastRecord, headerWeight, kwWeight, bIdfs, bIdTerms)
lastRecord = newLastRecord.toInt

newLastModelRecord = getLastModelRecord()

println(" next loop ")

Thread.sleep(3000)

}












